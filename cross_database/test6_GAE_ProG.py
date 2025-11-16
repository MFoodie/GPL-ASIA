import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv, InnerProductDecoder
from torch_geometric.utils import degree
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import random
import os
import time
import psutil
import torch.cuda as cuda

# 设置随机种子，确保实验可重复性
seed = 4321
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tc.manual_seed(seed)
if tc.cuda.is_available():
    tc.cuda.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
tc.backends.cudnn.deterministic = True
tc.backends.cudnn.benchmark = False

# 设备设置
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义数据集列表
datasets = ['Cora', 'PubMed', 'CiteSeer']
results = {}

# 定义 GAE 编码器
class GAEEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 定义 GAE 模型
class GAEModel(GAE):
    def __init__(self, nfeat, nclass, nhid, dropout):
        encoder = GAEEncoder(nfeat, nhid, dropout)
        decoder = InnerProductDecoder()
        super(GAEModel, self).__init__(encoder, decoder)
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.prompt_graph = None

    def forward(self, x, edge_index):
        if self.prompt_graph is not None:
            x = x + self.prompt_graph
        z = self.encode(x, edge_index)
        output = self.classifier(z)
        return F.log_softmax(output, dim=1)

    def set_prompt_graph(self, data, prompt_graph=None):
        if prompt_graph is None:
            prompt_graph = tc.mean(data.x, dim=0).repeat(data.num_nodes, 1)
            prompt_graph += tc.randn_like(prompt_graph) * 0.1
        self.prompt_graph = prompt_graph.to(device)
        print(f'ProG graph set with shape: {prompt_graph.shape}')

# 预训练阶段：对比损失
def train_with_contrastive_loss(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    perturbed_x1, perturbed_edge_index1 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    perturbed_x2, perturbed_edge_index2 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    output1 = model(perturbed_x1, perturbed_edge_index1)
    output2 = model(perturbed_x2, perturbed_edge_index2)
    loss = contrastive_loss(output1, output2)
    z = model.encode(perturbed_x1, perturbed_edge_index1)
    recon_loss = model.recon_loss(z, data.edge_index)
    total_loss = loss + recon_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# 对比损失计算
def contrastive_loss(x1, x2, temperature=0.1):
    x1_norm = x1.norm(dim=1, keepdim=True)
    x2_norm = x2.norm(dim=1, keepdim=True)
    sim_matrix = tc.matmul(x1, x2.T) / (x1_norm * x2_norm.T)
    sim_matrix = tc.exp(sim_matrix / temperature)
    pos_sim = sim_matrix.diag()
    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = -tc.log(loss).mean()
    return loss

# 微调阶段：标准训练
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    z = model.encode(data.x, data.edge_index)
    classification_loss = criterion(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    recon_loss = model.recon_loss(z, data.edge_index)
    total_loss = 0.5 * classification_loss + recon_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# CrossBA参数
reg_param_1 = 0.2  # 隐蔽性权重
reg_param_2 = 0.15  # 触发器一致性权重

# 生成目标嵌入
def generate_target_embedding(data, model, target_node):
    """生成目标嵌入用于CrossBA攻击"""
    # 创建触发器节点
    trigger_x = data.x[target_node:target_node+1].clone()
    trigger_edge_index = tc.tensor([[0], [0]], dtype=tc.long).to(data.x.device)
    
    # 获取目标嵌入
    with tc.no_grad():
        target_embedding = model(trigger_x, trigger_edge_index)
    
    return target_embedding

# CrossBA多目标特征扰动
def crossba_perturb_node_features(x, model, edge_index, perturbation_ratio, epsilon, target_embedding, reg_param_1, reg_param_2):
    """CrossBA多目标优化的特征扰动"""
    x = x.clone().detach().requires_grad_(True)
    
    # 前向传播
    output = model(x, edge_index)
    loss = F.cross_entropy(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    
    # 计算梯度
    model.zero_grad()
    loss.backward()
    grad = x.grad
    
    # 选择重要特征
    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio)
    important_features = tc.argsort(feature_importance)[-top_k_features:]
    
    # 计算相似性保持项
    with tc.no_grad():
        original_emb = model(x, edge_index)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        cos_sim_p = F.cosine_similarity(original_emb, target_embedding.expand(original_emb.size(0), -1), dim=1).mean()
    
    # 多目标扰动
    perturbed_x = x.clone()
    for idx in important_features:
        # 梯度引导扰动
        gradient_perturbation = epsilon * grad[:, idx].sign()
        # 相似性保持扰动
        similarity_perturbation = reg_param_1 * (1 - cos_sim_p) * tc.randn_like(grad[:, idx]) * 0.1
        # 随机噪声
        random_perturbation = reg_param_2 * tc.randn_like(grad[:, idx]) * epsilon * 0.5
        
        total_perturbation = gradient_perturbation + similarity_perturbation + random_perturbation
        perturbed_x[:, idx] += total_perturbation
    
    # 约束扰动范围
    perturbed_x = tc.clamp(perturbed_x, min=x.min(), max=x.max())
    
    return perturbed_x

# 自适应对抗攻击
def stealthy_adversarial_attack(data, model, epsilon=0.15, perturbation_ratio=0.4, task_type='node_classification'):
    model.eval()
    original_acc = test(model, data)
    perturbation_ratio = min(perturbation_ratio,
                             0.5 * (1 - original_acc) if task_type == 'node_classification' else 0.3)

    target_node = select_low_centrality_node(data, task_type)
    perturbed_edge_index = add_trigger_nodes(data, target_node,
                                             num_trigger_nodes=5 if task_type == 'node_classification' else 3)
    perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
    perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)

    # 生成目标嵌入用于CrossBA攻击
    target_embedding = generate_target_embedding(data, model, target_node)
    
    # 使用CrossBA多目标优化进行特征扰动
    perturbed_x = crossba_perturb_node_features(data.x, model, data.edge_index, 
                                               perturbation_ratio, epsilon, target_embedding, 
                                               reg_param_1, reg_param_2)

    max_attempts = 3
    attempt = 0
    perturbed_acc = test_after_attack(model, data, perturbed_x, perturbed_edge_index)
    while perturbed_acc > original_acc * 0.85 and attempt < max_attempts:
        perturbation_ratio = min(perturbation_ratio * 1.3, 0.6)
        epsilon = min(epsilon * 1.2, 0.3)
        perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
        perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)
        
        # 重新生成目标嵌入并应用CrossBA攻击
        target_embedding = generate_target_embedding(data, model, target_node)
        perturbed_x = crossba_perturb_node_features(data.x, model, data.edge_index, 
                                                   perturbation_ratio, epsilon, target_embedding, 
                                                   reg_param_1, reg_param_2)
        perturbed_acc = test_after_attack(model, data, perturbed_x, perturbed_edge_index)
        attempt += 1
        # 简化输出，只在最后一次尝试时显示
        if attempt == max_attempts - 1:


                    print(f"Attack Attempt {attempt}: Perturbed Acc = {perturbed_acc:.4f}, Perturbation Ratio = {perturbation_ratio:.4f}")

    return perturbed_x, perturbed_edge_index

# 选择低中心性节点
def select_low_centrality_node(data, task_type='node_classification'):
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    if task_type == 'node_classification':
        test_nodes = tc.where(data.test_mask.to(device))[0].to(device)
        if len(test_nodes) > 0:
            test_degrees = degrees[test_nodes]
            low_degree_nodes = test_nodes[tc.argsort(test_degrees)[:10]]
        else:
            low_degree_nodes = tc.argsort(degrees)[:10]
        class_counts = tc.bincount(data.y.to(device)[data.train_mask.to(device)])
        minority_class = tc.argmin(class_counts).item()
        minority_nodes = low_degree_nodes[data.y.to(device)[low_degree_nodes] == minority_class]
        if len(minority_nodes) > 0:
            return minority_nodes[np.random.randint(0, len(minority_nodes))].item()
        return low_degree_nodes[0].item()
    else:
        mid_degree_nodes = tc.argsort(degrees)[int(0.3 * len(degrees)):int(0.5 * len(degrees))]
        return mid_degree_nodes[np.random.randint(0, len(mid_degree_nodes))].item()

# 触发节点攻击
def add_trigger_nodes(data, target_node, num_trigger_nodes=5):
    num_nodes = data.x.size(0)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    low_degree_nodes = tc.argsort(degrees)[:num_trigger_nodes * 2]
    selected_nodes = np.random.choice(low_degree_nodes.cpu().numpy(), num_trigger_nodes, replace=False)
    edge_index = data.edge_index.clone().to(device)
    for node in selected_nodes:
        new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long).to(device)
        edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index

# 触发诱导子图攻击
def add_trigger_induced_subgraph(data, perturbation_ratio=0.2, task_type='node_classification'):
    num_nodes = data.x.size(0)
    edge_density = data.edge_index.size(1) / (num_nodes * (num_nodes - 1))
    perturbation_ratio = min(perturbation_ratio,
                             0.1 / edge_density if task_type == 'node_classification' else 0.5 / edge_density)
    subgraph_size = int(num_nodes * perturbation_ratio)

    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    test_nodes = tc.where(data.test_mask.to(device))[0].to(device)
    if len(test_nodes) > 0:
        test_degrees = degrees[test_nodes]
        perturb_nodes = test_nodes[tc.argsort(test_degrees)[:subgraph_size]].cpu().numpy()
    else:
        perturb_nodes = np.random.choice(num_nodes, subgraph_size, replace=False)

    edge_index = data.edge_index.clone().to(device)
    low_degree_nodes = tc.argsort(degrees)[:int(num_nodes * 0.3)].cpu().numpy()
    num_edges = 3 if task_type == 'node_classification' else 2
    for node in perturb_nodes:
        for _ in range(num_edges):
            target_node = np.random.choice(low_degree_nodes)
            new_edges = tc.tensor([[node, target_node], [target_node, node]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)

    return edge_index

# 结构扰动
def structure_perturbation(data, edge_index, perturbation_ratio=0.2):
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)

    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    edge_importance = degrees[edge_index[0]] + degrees[edge_index[1]]
    low_importance_edges = tc.argsort(edge_importance)[:num_perturbations]

    remaining_edges = np.delete(edge_index.cpu().numpy(), low_importance_edges.cpu().numpy(), axis=1)
    remaining_edges = tc.tensor(remaining_edges, dtype=tc.long).to(device)

    new_edges = set()
    test_nodes = tc.where(data.test_mask.to(device))[0].to(device).cpu().numpy()
    low_degree_nodes = tc.argsort(degrees)[:int(data.num_nodes * 0.3)].cpu().numpy()
    while len(new_edges) < num_perturbations:
        u = np.random.choice(test_nodes if len(test_nodes) > 0 else low_degree_nodes)
        v = np.random.choice(low_degree_nodes)
        if u != v:
            new_edges.add((min(u, v), max(u, v)))
    new_edges = np.array(list(new_edges)).T
    perturbed_edge_index = tc.cat([remaining_edges, tc.tensor(new_edges, dtype=tc.long).to(device)], dim=1)

    return perturbed_edge_index

# 特征扰动
def perturb_node_features(x, model, perturbation_ratio=0.2, epsilon=0.15, task_type='node_classification'):
    x = x.clone().detach().requires_grad_(True).to(device)
    output = model(x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    model.zero_grad()
    loss.backward()
    grad = x.grad

    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio * 1.5)
    important_features = tc.argsort(feature_importance)[-top_k_features:]

    feature_variance = tc.var(x, dim=0)
    adaptive_epsilon = epsilon * feature_variance / (feature_variance.mean() + 1e-8)
    if task_type == 'link_prediction':
        adaptive_epsilon *= 0.7

    perturbed_x = x.clone()
    for idx in important_features:
        perturbed_x[:, idx] += adaptive_epsilon[idx] * grad[:, idx].sign()

    perturbed_x = tc.clamp(perturbed_x, min=x.min(), max=x.max())

    mean_diff = (x.mean() - perturbed_x.mean()).abs().item()
    var_diff = (x.var() - perturbed_x.var()).abs().item()
    if mean_diff > 0.5 or var_diff > 0.5:
        print(f"警告: 特征分布变化较大 (Mean Diff: {mean_diff:.4f}, Var Diff: {var_diff:.4f})")

    return perturbed_x

# 测试模型
def test(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
    acc = int(correct) / int(data.test_mask.to(device).sum())
    return acc

# 执行攻击并微调
def perform_attack_and_finetune(model, data, optimizer, criterion):
    perturbed_x, perturbed_edge_index = stealthy_adversarial_attack(data, model)
    for epoch in range(100):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            acc = test(model, data)
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    return perturbed_x, perturbed_edge_index

# 计算置信度
def calculate_confidence(model, data, mask):
    output = model(data.x, data.edge_index)
    prob = F.softmax(output[mask], dim=1)
    max_probs = prob.max(dim=1)[0]
    confidence = max_probs.mean().item()
    return confidence

# 评估指标
def evaluate_with_metrics(model, data, perturbed_x, perturbed_edge_index):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index)
    y_pred_pos = F.softmax(output[data.test_mask.to(device)], dim=1).cpu().detach().numpy().max(axis=1)
    y_pred_neg = F.softmax(output[~data.test_mask.to(device)], dim=1).cpu().detach().numpy().max(axis=1)

    if len(y_pred_pos) != len(y_pred_neg):
        print(f"警告: 正样本数量 ({len(y_pred_pos)}) 和负样本数量 ({len(y_pred_neg)}) 不一致，进行下采样对齐。")
        min_samples = min(len(y_pred_pos), len(y_pred_neg))
        y_pred_pos = np.random.choice(y_pred_pos, min_samples, replace=False)
        y_pred_neg = np.random.choice(y_pred_neg, min_samples, replace=False)

    class Evaluator:
        def __init__(self, eval_metric):
            self.eval_metric = eval_metric

        def eval(self, input_dict):
            y_pred_pos = input_dict['y_pred_pos']
            y_pred_neg = input_dict['y_pred_neg']
            if self.eval_metric == 'hits@50':
                threshold = np.sort(y_pred_neg)[-50] if len(y_pred_neg) >= 50 else np.max(y_pred_neg)
                return {'hits@50': np.mean(y_pred_pos > threshold)}
            elif self.eval_metric == 'mrr':
                ranks = np.sum(y_pred_neg > y_pred_pos[:, None], axis=1) + 1
                return {'mrr': np.mean(1.0 / ranks)}

    evaluator_hits = Evaluator(eval_metric='hits@50')
    evaluator_mrr = Evaluator(eval_metric='mrr')
    hits_at_k = evaluator_hits.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    mrr_score = evaluator_mrr.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    print(f"Hits@50: {hits_at_k['hits@50']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score['mrr']:.4f}")
    return hits_at_k, mrr_score

# 攻击后测试
def test_after_attack(model, data, perturbed_x, perturbed_edge_index):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
    acc = int(correct) / int(data.test_mask.to(device).sum())
    return acc

# 计算攻击成功率
def calculate_asr(model, data, perturbed_x, perturbed_edge_index):
    original_pred = model(data.x, data.edge_index).argmax(dim=1)
    perturbed_pred = model(perturbed_x, perturbed_edge_index).argmax(dim=1)
    # 修复ASR计算：比较原始预测和攻击后预测的差异
    successful_attack = (perturbed_pred[data.test_mask.to(device)] != original_pred[data.test_mask.to(device)]).sum().item()
    total_attack = data.test_mask.to(device).sum().item()
    asr = successful_attack / total_attack if total_attack > 0 else 0
    return asr

# 计算特征扰动比例
def calculate_feature_perturbation_ratio(original_x, perturbed_x):
    perturbation = (original_x - perturbed_x).abs()
    feature_perturbation_ratio = perturbation.mean().item()
    return feature_perturbation_ratio

# 攻击指标评估
def evaluate_attack_metrics(model, data, perturbed_x, perturbed_edge_index, dataset_name):
    try:
        original_pred = model(data.x, data.edge_index).argmax(dim=1)
        perturbed_pred = model(perturbed_x, perturbed_edge_index).argmax(dim=1)

        asr = calculate_asr(model, data, perturbed_x, perturbed_edge_index)
        
        # 添加ASR等级评判标准
        if asr >= 0.8:
            asr_level = "优秀 (Excellent)"
        elif asr >= 0.6:
            asr_level = "良好 (Good)"
        elif asr >= 0.4:
            asr_level = "一般 (Fair)"
        elif asr >= 0.2:
            asr_level = "较差 (Poor)"
        else:
            asr_level = "很差 (Very Poor)"
        
        print(f'Attack Success Rate (ASR): {asr:.4f} - {asr_level}')

        feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
        print(f'Perturbation Ratio: {feature_perturbation_ratio:.4f}')

        original_confidence = calculate_confidence(model, data, data.test_mask.to(device))
        perturbed_confidence = calculate_confidence(model, data, data.test_mask.to(device))
        confidence_drop = original_confidence - perturbed_confidence
        print(f'Original Confidence: {original_confidence:.4f}')
        print(f'Perturbed Confidence: {perturbed_confidence:.4f}')
        print(f'Confidence Drop: {confidence_drop:.4f}')

        y_true = data.y.to(device)[data.test_mask.to(device)].cpu().numpy()
        y_pred = perturbed_pred[data.test_mask.to(device)].cpu().numpy()

        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"警告: 数据集 {dataset_name} 的测试集为空，无法计算评估指标")
            return asr

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        nclass = tc.unique(data.y.to(device)).size(0)
        class_names = [str(i) for i in range(nclass)]
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='inferno_r', xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{dataset_name} - Confusion Matrix')
        plt.show()
        plt.close()

        y_pred_probs = F.softmax(model(perturbed_x, perturbed_edge_index)[data.test_mask.to(device)], dim=1).detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
        print(f'ROC-AUC: {auc:.4f}')

        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        for i in range(nclass):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if tn + fp > 0 else 0
            specificity_per_class.append(specificity)
        specificity = np.mean(specificity_per_class)
        print(f'Specificity: {specificity:.4f}')

        return asr
    except Exception as e:
        print(f"数据集 {dataset_name} 的攻击指标评估失败: {str(e)}")
        return 0

# 资源使用评估
def evaluate_resource_usage(model, data, perturbed_x, perturbed_edge_index, dataset_name):
    gpu_mem_usage = []
    cpu_mem_usage = []
    time_delays = []

    if tc.cuda.is_available():
        tc.cuda.empty_cache()

    print(f"\n评估 {dataset_name} 原始模型资源使用...")
    start_time = time.time()
    with tc.no_grad():
        model.eval()
        original_output = model(data.x, data.edge_index)
    original_time = time.time() - start_time

    if tc.cuda.is_available():
        gpu_mem = cuda.memory_allocated() / (1024 ** 2)
        gpu_mem_usage.append(gpu_mem)
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
    cpu_mem_usage.append(cpu_mem)
    time_delays.append(original_time)

    print(f"评估 {dataset_name} 攻击后模型资源使用...")
    start_time = time.time()
    with tc.no_grad():
        perturbed_output = model(perturbed_x, perturbed_edge_index)
    attack_time = time.time() - start_time

    if tc.cuda.is_available():
        gpu_mem = cuda.memory_allocated() / (1024 ** 2)
        gpu_mem_usage.append(gpu_mem)
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
    cpu_mem_usage.append(cpu_mem)
    time_delays.append(attack_time)

    print(f"评估 {dataset_name} 训练过程资源使用...")
    train_mem_usage = []
    train_time_usage = []
    for epoch in range(5):
        start_time = time.time()
        train_loss = train(model, data, optimizer, criterion)
        epoch_time = time.time() - start_time
        if tc.cuda.is_available():
            gpu_mem = cuda.memory_allocated() / (1024 ** 2)
            train_mem_usage.append(gpu_mem)
        train_time_usage.append(epoch_time)

    print(f"\n=== {dataset_name} 资源使用评估报告 ===")
    print(f"原始模型推理时间: {original_time:.4f}s")
    print(f"攻击后推理时间: {attack_time:.4f}s")
    print(f"平均训练时间/epoch: {np.mean(train_time_usage):.4f}s")

    if tc.cuda.is_available():
        print(f"\nGPU内存使用:")
        print(f"- 原始模型: {gpu_mem_usage[0]:.2f} MB")
        print(f"- 攻击后模型: {gpu_mem_usage[1]:.2f} MB")
        print(f"- 训练峰值内存: {max(train_mem_usage):.2f} MB")

    print(f"\nCPU内存使用:")
    print(f"- 原始模型: {cpu_mem_usage[0]:.2f} MB")
    print(f"- 攻击后模型: {cpu_mem_usage[1]:.2f} MB")

    plt.figure(figsize=(15, 5))
    if tc.cuda.is_available():
        plt.subplot(1, 3, 1)
        plt.bar(['Original', 'Attacked'], gpu_mem_usage)
        plt.title(f'{dataset_name} GPU Memory Usage (MB)')
        plt.ylabel('Memory (MB)')

    plt.subplot(1, 3, 2)
    plt.bar(['Original', 'Attacked'], cpu_mem_usage)
    plt.title(f'{dataset_name} CPU Memory Usage (MB)')
    plt.ylabel('Memory (MB)')

    plt.subplot(1, 3, 3)
    plt.bar(['Original', 'Attacked'], time_delays)
    plt.title(f'{dataset_name} Inference Time (s)')
    plt.ylabel('Time (seconds)')

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))
    if tc.cuda.is_available():
        plt.subplot(1, 2, 1)
        plt.plot(train_mem_usage)
        plt.title(f'{dataset_name} Training GPU Memory Usage')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')

    plt.subplot(1, 2, 2)
    plt.plot(train_time_usage)
    plt.title(f'{dataset_name} Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')

    plt.tight_layout()
    plt.show()
    plt.close()

# 计算性能分析
def profile_computation(model, data, perturbed_x, perturbed_edge_index, dataset_name):
    print(f"\n运行 {dataset_name} 计算性能分析...")
    with tc.profiler.profile(
            activities=[tc.profiler.ProfilerActivity.CPU, tc.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_flops=True
    ) as prof:
        with tc.no_grad():
            model(data.x, data.edge_index)
            model(perturbed_x, perturbed_edge_index)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(f"{dataset_name}_performance_trace.json")
    print(f"性能分析跟踪已保存到 {dataset_name}_performance_trace.json")

# 跨数据集测试主程序
if __name__ == "__main__":
    for dataset_name in datasets:
        print(f"\n=== 开始测试数据集: {dataset_name} ===")

        # 加载数据集
        try:
            dataset = Planetoid(root='D:/SRTP/SRTP/data', name=dataset_name, force_reload=False)
            data = dataset[0].to(device)
        except Exception as e:
            print(f"加载数据集 {dataset_name} 失败: {str(e)}")
            continue

        # 初始化模型
        nfeat = dataset.num_features
        nclass = tc.unique(data.y.to(device)).size(0)
        nhid = 16
        dropout = 0.6
        model = GAEModel(nfeat, nclass, nhid, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # 设置提示图
        model.set_prompt_graph(data)

        try:
            # 预训练
            for epoch in range(100):
                loss = train_with_contrastive_loss(model, data, optimizer)
                if epoch % 10 == 0:
                    print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

            # 执行攻击并微调
            perturbed_x, perturbed_edge_index = perform_attack_and_finetune(model, data, optimizer, criterion)
            original_acc = test(model, data)
            print(f'Original Accuracy (Before Attack): {original_acc:.4f}')
            perturbed_acc = test_after_attack(model, data, perturbed_x, perturbed_edge_index)
            print(f'Perturbed Accuracy (After Attack): {perturbed_acc:.4f}')

            # 评估攻击指标
            asr = evaluate_attack_metrics(model, data, perturbed_x, perturbed_edge_index, dataset_name)

            # 评估其他指标
            hits_at_k, mrr_score = evaluate_with_metrics(model, data, perturbed_x, perturbed_edge_index)

            # 资源使用评估
            evaluate_resource_usage(model, data, perturbed_x, perturbed_edge_index, dataset_name)

            # 计算性能分析（仅在 GPU 可用时执行）
            if tc.cuda.is_available():
                        profile_computation(model, data, perturbed_x, perturbed_edge_index, dataset_name)

            # 存储结果
            results[dataset_name] = {
                'original_acc': original_acc,
                'perturbed_acc': perturbed_acc,
                'asr': asr,
                'hits_at_k': hits_at_k['hits@50'],
                'mrr': mrr_score['mrr']
            }

        except Exception as e:
            print(f"数据集 {dataset_name} 处理时发生错误: {str(e)}")
            continue

    # 打印所有数据集的结果
    print("\n=== 跨数据集测试结果总结 ===")
    for dataset_name in results:
        print(f"\n数据集: {dataset_name}")
        print(f"原始准确率: {results[dataset_name]['original_acc']:.4f}")
        print(f"攻击后准确率: {results[dataset_name]['perturbed_acc']:.4f}")
        print(f"攻击成功率 (ASR): {results[dataset_name]['asr']:.4f}")
        print(f"Hits@50: {results[dataset_name]['hits_at_k']:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {results[dataset_name]['mrr']:.4f}")
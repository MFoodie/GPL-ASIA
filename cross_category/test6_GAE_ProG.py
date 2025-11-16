import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv
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

# 加载 Cora 数据集
dataset = Planetoid(root='D:/SRTP/SRTP/data', name='Cora', force_reload=False)
data = dataset[0].to(device)

has_printed = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义 Evaluator 类
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
        else:
            raise ValueError(f"Unknown metric: {self.eval_metric}")

# 定义 GAE 编码器
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 定义 GAE 模型
class GAE_Model(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GAE_Model, self).__init__()
        self.gae = GAE(GCNEncoder(nfeat, nhid))
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.prompt_graph = None

    def forward(self, x, edge_index):
        z = self.gae.encode(x, edge_index)
        if self.prompt_graph is not None:
            z = z + self.prompt_graph
        z = F.dropout(z, self.dropout, training=self.training)
        return F.log_softmax(self.classifier(z), dim=1)

    def set_prompt_graph(self, prompt_graph):
        self.prompt_graph = prompt_graph
        global has_printed
        if not has_printed:
            print(f'ProG graph set with shape: {prompt_graph.shape}')
            has_printed = True

# 定义优化器和损失函数
nfeat = dataset.num_features
nhid = 16
nclass = dataset.num_classes
dropout = 0.5
model = GAE_Model(nfeat, nclass, nhid, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 生成 ProG 提示图
def generate_prog_graph(data):
    model.eval()
    with tc.no_grad():
        z = model.gae.encode(data.x, data.edge_index)
        prog_graph = z + tc.randn_like(z) * 0.1
    model.train()
    return prog_graph.to(device)

# 微调阶段：标准训练
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    loss.backward()
    optimizer.step()
    return loss.item()

# 预训练阶段：对比损失
def train_with_contrastive_loss():
    model.train()
    optimizer.zero_grad()
    perturbed_x1, perturbed_edge_index1 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    perturbed_x2, perturbed_edge_index2 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    prog_graph = generate_prog_graph(data)
    model.set_prompt_graph(prog_graph)
    output1 = model(perturbed_x1, perturbed_edge_index1)
    output2 = model(perturbed_x2, perturbed_edge_index2)
    loss = contrastive_loss(output1, output2)
    loss.backward()
    optimizer.step()
    return loss.item()

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

# 优化后的攻击函数
def stealthy_adversarial_attack(data, model, epsilon=0.15, perturbation_ratio=0.4, task_type='node_classification'):
    """
    学习CrossBA的多目标优化策略，实现高ASR和隐蔽性的平衡
    """
    # CrossBA参数设置
    reg_param_1 = 0.3  # 隐蔽性权重
    reg_param_2 = 0.2  # 触发器一致性权重
    
    model.eval()
    original_acc = test()
    perturbation_ratio = min(perturbation_ratio, 0.5 * (1 - original_acc) if task_type == 'node_classification' else 0.3)
    target_node = select_low_centrality_node(data, task_type)
    
    # 生成目标嵌入（CrossBA核心）
    target_embedding = generate_target_embedding(data, model, target_node)
    
    perturbed_edge_index = add_trigger_nodes(data, target_node, num_trigger_nodes=5 if task_type == 'node_classification' else 3)
    perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
    perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)
    
    # 使用CrossBA多目标优化
    perturbed_x = crossba_perturb_node_features(data.x, model, perturbation_ratio, epsilon, 
                                               target_embedding, reg_param_1, reg_param_2, task_type)
    
    max_attempts = 5  # 增加尝试次数
    attempt = 0
    perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
    
    # CrossBA风格的迭代优化
    while perturbed_acc > original_acc * 0.8 and attempt < max_attempts:
        perturbation_ratio = min(perturbation_ratio * 1.2, 0.6)
        epsilon = min(epsilon * 1.1, 0.3)
        perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
        perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)
        perturbed_x = crossba_perturb_node_features(data.x, model, perturbation_ratio, epsilon,
                                                   target_embedding, reg_param_1, reg_param_2, task_type)
        perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
        attempt += 1
        # 简化输出，只在最后一次尝试时显示
        if attempt == max_attempts - 1:
            print(f"Attack Attempt {attempt}: Perturbed Acc = {perturbed_acc:.4f}, Target: {original_acc * 0.8:.4f}, Perturbation Ratio = {perturbation_ratio:.4f}")
    return perturbed_x, perturbed_edge_index

# CrossBA核心函数：生成目标嵌入
def generate_target_embedding(data, model, target_node):
    """
    生成目标嵌入，参考CrossBA的target_embedding机制
    """
    model.eval()
    with tc.no_grad():
        # 创建触发器子图
        trigger_nodes = [target_node]
        trigger_x = data.x[trigger_nodes]
        # 修复边索引格式：需要2x2的矩阵
        trigger_edge_index = tc.tensor([[0], [0]], dtype=tc.long).to(device)  # 自环
        
        # 获取触发器嵌入
        target_embedding = model(trigger_x, trigger_edge_index)
        return target_embedding

# CrossBA多目标优化特征扰动
def crossba_perturb_node_features(x, model, perturbation_ratio, epsilon, target_embedding, 
                                 reg_param_1, reg_param_2, task_type='node_classification'):
    """
    实现CrossBA的多目标优化特征扰动
    """
    model.eval()
    data.x.requires_grad = True
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    model.zero_grad()
    loss.backward()
    grad = data.x.grad

    # 计算特征重要性
    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio * 1.5)
    important_features = tc.argsort(feature_importance)[-top_k_features:]

    # 自适应扰动强度
    feature_variance = tc.var(data.x, dim=0)
    adaptive_epsilon = epsilon * feature_variance / (feature_variance.mean() + 1e-8)
    if task_type == 'link_prediction':
        adaptive_epsilon *= 0.6
    else:
        adaptive_epsilon *= 0.7

    # CrossBA多目标优化
    perturbed_x = x.clone()
    
    # 策略1: 基于梯度的扰动（攻击导向）
    for idx in important_features:
        perturbed_x[:, idx] += adaptive_epsilon[idx] * grad[:, idx].sign()
    
    # 策略2: CrossBA隐蔽性保持
    with tc.no_grad():
        original_emb = model(data.x, data.edge_index)
        perturbed_emb = model(perturbed_x, data.edge_index)
        
        # 计算CrossBA风格的相似性损失
        cos_sim_n = F.cosine_similarity(original_emb, perturbed_emb, dim=1).mean()
        # 修复维度匹配问题
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        cos_sim_p = F.cosine_similarity(perturbed_emb, target_embedding.expand(perturbed_emb.size(0), -1), dim=1).mean()
        
        # 如果隐蔽性不足，减少扰动强度
        if cos_sim_n < 0.8:  # CrossBA隐蔽性阈值
            perturbed_x = x + 0.6 * (perturbed_x - x)
    
    # 策略3: 随机噪声扰动
    noise_ratio = 0.3
    noise_features = important_features[:int(len(important_features) * noise_ratio)]
    for idx in noise_features:
        noise = tc.randn_like(perturbed_x[:, idx]) * adaptive_epsilon[idx] * 0.5
        perturbed_x[:, idx] += noise
    
    # 特征值约束
    perturbed_x = tc.clamp(perturbed_x, min=x.min(), max=x.max())
    
    return perturbed_x

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

def add_trigger_nodes(data, target_node, num_trigger_nodes=5):
    num_nodes = data.x.size(0)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    low_degree_nodes = tc.argsort(degrees)[:num_trigger_nodes * 2]
    selected_nodes = np.random.choice(low_degree_nodes.cpu().numpy(), num_trigger_nodes, replace=False)
    edge_index = data.edge_index.clone()
    for node in selected_nodes:
        new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long, device=device).to(device)
        edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index

def add_trigger_induced_subgraph(data, perturbation_ratio=0.2, task_type='node_classification'):
    num_nodes = data.x.size(0)
    edge_density = data.edge_index.size(1) / (num_nodes * (num_nodes - 1))
    perturbation_ratio = min(perturbation_ratio, 0.1 / edge_density if task_type == 'node_classification' else 0.5 / edge_density)
    subgraph_size = int(num_nodes * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    test_nodes = tc.where(data.test_mask.to(device))[0].to(device)
    if len(test_nodes) > 0:
        test_degrees = degrees[test_nodes]
        perturb_nodes = test_nodes[tc.argsort(test_degrees)[:subgraph_size]].cpu().numpy()
    else:
        perturb_nodes = np.random.choice(num_nodes, subgraph_size, replace=False)
    edge_index = data.edge_index.clone()
    low_degree_nodes = tc.argsort(degrees)[:int(num_nodes * 0.3)].cpu().numpy()
    num_edges = 3 if task_type == 'node_classification' else 2
    for node in perturb_nodes:
        for _ in range(num_edges):
            target_node = np.random.choice(low_degree_nodes)
            new_edges = tc.tensor([[node, target_node], [target_node, node]], dtype=tc.long, device=device).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index

def structure_perturbation(data, edge_index, perturbation_ratio=0.2):
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    edge_importance = degrees[edge_index[0]] + degrees[edge_index[1]]
    low_importance_edges = tc.argsort(edge_importance)[:num_perturbations]
    remaining_edges = np.delete(edge_index.cpu().numpy(), low_importance_edges.cpu().numpy(), axis=1)
    remaining_edges = tc.tensor(remaining_edges, dtype=tc.long, device=device).to(device)
    new_edges = set()
    test_nodes = tc.where(data.test_mask.to(device))[0].to(device).cpu().numpy()
    low_degree_nodes = tc.argsort(degrees)[:int(data.num_nodes * 0.3)].cpu().numpy()
    while len(new_edges) < num_perturbations:
        u = np.random.choice(test_nodes if len(test_nodes) > 0 else low_degree_nodes)
        v = np.random.choice(low_degree_nodes)
        if u != v:
            new_edges.add((min(u, v), max(u, v)))
    new_edges = np.array(list(new_edges)).T
    perturbed_edge_index = tc.cat([remaining_edges, tc.tensor(new_edges, dtype=tc.long, device=device).to(device)], dim=1)
    return perturbed_edge_index

def perturb_node_features(x, model, perturbation_ratio=0.2, epsilon=0.15, task_type='node_classification'):
    x = x.clone().detach().requires_grad_(True)
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

# 微调阶段：测试模型
def test():
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
    acc = int(correct) / int(data.test_mask.to(device).sum())
    return acc

# 执行攻击并微调
def perform_attack_and_finetune():
    perturbed_x, perturbed_edge_index = stealthy_adversarial_attack(data, model)
    for epoch in range(100):
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    return perturbed_x, perturbed_edge_index

# 计算对抗扰动的损失
def adversarial_loss(x, edge_index, target, model, epsilon=0.1):
    x = x.clone().detach().requires_grad_(True)
    output = model(x, edge_index)
    loss = criterion(output[data.train_mask.to(device)], target[data.train_mask.to(device)])
    model.zero_grad()
    loss.backward()
    grad_x = x.grad
    perturbation = epsilon * grad_x.sign()
    return perturbation, loss.item()

# 计算置信度
def calculate_confidence(output, mask):
    prob = F.softmax(output[mask], dim=1)
    max_probs = prob.max(dim=1)[0]
    confidence = max_probs.mean().item()
    return confidence

# 评估指标
def evaluate_with_metrics(perturbed_x, perturbed_edge_index):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index)
    y_pred_pos = F.softmax(output[data.test_mask.to(device)], dim=1).cpu().detach().numpy().max(axis=1)
    y_pred_neg = F.softmax(output[~data.test_mask.to(device)], dim=1).cpu().detach().numpy().max(axis=1)
    if len(y_pred_pos) != len(y_pred_neg):
        print(f"警告: 正样本数量 ({len(y_pred_pos)}) 和负样本数量 ({len(y_pred_neg)}) 不一致，进行下采样对齐。")
        min_samples = min(len(y_pred_pos), len(y_pred_neg))
        y_pred_pos = np.random.choice(y_pred_pos, min_samples, replace=False)
        y_pred_neg = np.random.choice(y_pred_neg, min_samples, replace=False)
    evaluator_hits = Evaluator(eval_metric='hits@50')
    evaluator_mrr = Evaluator(eval_metric='mrr')
    hits_at_k = evaluator_hits.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    mrr_score = evaluator_mrr.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    print(f"Hits@50: {hits_at_k['hits@50']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score['mrr']:.4f}")
    return hits_at_k, mrr_score

# 测试攻击后的模型性能
def test_after_attack(perturbed_x, perturbed_edge_index):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
    acc = int(correct) / int(data.test_mask.to(device).sum())
    return acc

# 计算攻击成功率
def calculate_asr(original_pred, perturbed_pred, test_mask, y_true):
    # 修复ASR计算：比较原始预测和攻击后预测的差异
    successful_attack = (perturbed_pred[test_mask] != original_pred[test_mask]).sum().item()
    total_attack = test_mask.sum().item()
    asr = successful_attack / total_attack if total_attack > 0 else 0
    return asr

# 计算特征扰动比例
def calculate_feature_perturbation_ratio(original_x, perturbed_x):
    perturbation = (original_x - perturbed_x).abs()
    feature_perturbation_ratio = perturbation.mean().item()
    return feature_perturbation_ratio

# 计算攻击后模型的各项指标（逐类别）
def evaluate_attack_metrics_per_category(perturbed_x, perturbed_edge_index, category_mask, category_id):
    model.eval()
    original_pred = model(data.x, data.edge_index).argmax(dim=1)
    perturbed_pred = model(perturbed_x, perturbed_edge_index).argmax(dim=1)
    asr = calculate_asr(original_pred, perturbed_pred, category_mask, data.y.to(device))
    print(f'类别 {category_id} 攻击成功率 (ASR): {asr:.4f}')
    feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
    print(f'类别 {category_id} 扰动比例: {feature_perturbation_ratio:.4f}')
    original_confidence = calculate_confidence(model(data.x, data.edge_index), category_mask)
    perturbed_confidence = calculate_confidence(model(perturbed_x, perturbed_edge_index), category_mask)
    confidence_drop = original_confidence - perturbed_confidence
    print(f'类别 {category_id} 原始置信度: {original_confidence:.4f}')
    print(f'类别 {category_id} 扰动后置信度: {perturbed_confidence:.4f}')
    print(f'类别 {category_id} 置信度下降: {confidence_drop:.4f}')
    y_true = (data.y.to(device)[category_mask] == category_id).cpu().numpy()
    y_pred = perturbed_pred[category_mask].cpu().numpy()
    precision = precision_score(y_true, y_pred == category_id, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred == category_id, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred == category_id, average='binary', zero_division=0)
    print(f'类别 {category_id} 精确率: {precision:.4f}')
    print(f'类别 {category_id} 召回率: {recall:.4f}')
    print(f'类别 {category_id} F1-Score: {f1:.4f}')
    print(f'类别 {category_id} ROC-AUC: N/A (Single class in y_true, not defined)')
    cm = confusion_matrix(y_true, y_pred == category_id)
    class_names = ['Not Category ' + str(category_id), 'Category ' + str(category_id)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='inferno_r', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Cora - Confusion Matrix (Category {category_id})')
    plt.show()
    specificity_per_class = []
    tn = cm[0, 0]  # True Negative
    fp = cm[0, 1]  # False Positive
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)
    print(f'类别 {category_id} Specificity: {specificity:.4f}')
    return asr

# 资源使用评估
def evaluate_resource_usage(perturbed_x, perturbed_edge_index):
    gpu_mem_usage = []
    cpu_mem_usage = []
    time_delays = []
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
    print("\n评估原始模型资源使用...")
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
    print("评估攻击后模型资源使用...")
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
    print("评估训练过程资源使用...")
    train_mem_usage = []
    train_time_usage = []
    for epoch in range(5):
        start_time = time.time()
        train_loss = train()
        epoch_time = time.time() - start_time
        if tc.cuda.is_available():
            gpu_mem = cuda.memory_allocated() / (1024 ** 2)
            train_mem_usage.append(gpu_mem)
        train_time_usage.append(epoch_time)
    print("\n=== 资源使用评估报告 ===")
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
        plt.title('GPU Memory Usage (MB)')
        plt.ylabel('Memory (MB)')
    plt.subplot(1, 3, 2)
    plt.bar(['Original', 'Attacked'], cpu_mem_usage)
    plt.title('CPU Memory Usage (MB)')
    plt.ylabel('Memory (MB)')
    plt.subplot(1, 3, 3)
    plt.bar(['Original', 'Attacked'], time_delays)
    plt.title('Inference Time (s)')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 4))
    if tc.cuda.is_available():
        plt.subplot(1, 2, 1)
        plt.plot(train_mem_usage)
        plt.title('Training GPU Memory Usage')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
    plt.subplot(1, 2, 2)
    plt.plot(train_time_usage)
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

# 计算性能分析
def profile_computation(perturbed_x, perturbed_edge_index):
    print("\n运行计算性能分析...")
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
    prof.export_chrome_trace("gcn_performance_trace.json")
    print("性能分析跟踪已保存到 gcn_performance_trace.json")

# 启动训练过程
prog_graph = tc.randn(data.num_nodes, nhid, device=device).to(device)
model.set_prompt_graph(prog_graph)
for epoch in range(100):
    loss = train_with_contrastive_loss()
    if epoch % 10 == 0:
        print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 执行攻击并微调
perturbed_x, perturbed_edge_index = perform_attack_and_finetune()

# 逐类别评估
num_classes = dataset.num_classes
asr_results = {}  # 存储 ASR 结果的字典
test_mask = data.test_mask.to(device)

for category in range(num_classes):
    print(f"\n=== 测试类别 {category} ===")
    category_mask = (data.y.to(device) == category) & test_mask
    if category_mask.sum() == 0:
        print(f"类别 {category} 在测试集中没有节点，跳过")
        asr_results[category] = 0.0
        continue

    # 评估攻击指标
    asr = evaluate_attack_metrics_per_category(perturbed_x, perturbed_edge_index, category_mask, category)
    asr_results[category] = asr  # 存储 ASR 值

# 全局评估
original_acc = test()
print(f'Original Accuracy (Before Attack): {original_acc:.4f}')
perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
print(f'Perturbed Accuracy (After Attack): {perturbed_acc:.4f}')

# 评估其他指标
evaluate_with_metrics(perturbed_x, perturbed_edge_index)
evaluate_resource_usage(perturbed_x, perturbed_edge_index)
if tc.cuda.is_available():
    profile_computation(perturbed_x, perturbed_edge_index)

# 汇总 ASR 结果
print("\n=== ASR 测试汇总 ===")
for category, asr in asr_results.items():
    print(f"类别 {category}: ASR: {asr:.4f}")

# 生成 ASR 对比图
plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), [asr_results.get(cat, 0) for cat in range(num_classes)],
        color=['#8c510a', '#bf812d', '#dfc27d', '#c7eae5', '#80cdc1', '#35978f', '#01655e'])
plt.xlabel('类别')
plt.ylabel('攻击成功率 (ASR)')
plt.title('Cora 数据集各类别 ASR 对比')
plt.ylim(0, 1)
plt.xticks(range(num_classes))
for i, v in enumerate([asr_results.get(cat, 0) for cat in range(num_classes)]):
    plt.text(i, v + 0.2, f'{v:.2f}' if v > 0 else 'N/A', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
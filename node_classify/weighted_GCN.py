import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import random
import os
from evaluator import Evaluator
import time
import psutil
import torch.cuda as cuda

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 加载 Cora 数据集
dataset = Planetoid(root='D:/SRTP/SRTP/data', name='Cora', force_reload=False)
data = dataset[0]

# 计算加权图的边权重（基于节点特征的余弦相似度）
def compute_edge_weights(x, edge_index):
    num_edges = edge_index.size(1)
    edge_weight = tc.ones(num_edges, dtype=tc.float)
    for i in range(num_edges):
        src, dst = edge_index[:, i]
        feat_src = x[src]
        feat_dst = x[dst]
        cos_sim = F.cosine_similarity(feat_src.unsqueeze(0), feat_dst.unsqueeze(0)).item()
        edge_weight[i] = max(cos_sim, 1e-6)
    edge_weight = edge_weight / edge_weight.sum()
    return edge_weight.to(x.device)

# 初始化边权重
data.edge_weight = compute_edge_weights(data.x, data.edge_index)

# 定义加权 GCN 模型
class WeightedGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(WeightedGCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.prompt_graph = None
        self.nhid = nhid
        self.nclass = nclass

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        if self.prompt_graph is not None:
            prompt_projected = self.project_prompt(self.prompt_graph, x.size(1))
            x = x + prompt_projected
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def project_prompt(self, prompt_graph, output_dim):
        projection = nn.Linear(prompt_graph.size(1), output_dim).to(prompt_graph.device)
        return projection(prompt_graph)

    def set_prompt_graph(self, prompt_graph):
        self.prompt_graph = prompt_graph
        print(f'提示图已设置，形状: {prompt_graph.shape}')

# 定义优化器和损失函数
nfeat = dataset.num_features
nhid = 8
nclass = dataset.num_classes
dropout = 0.6
model = WeightedGCN(nfeat, nclass, nhid, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 预训练阶段：对比损失
def train_with_contrastive_loss():
    model.train()
    optimizer.zero_grad()
    perturbed_x1, perturbed_edge_index1, perturbed_weight1 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    perturbed_x2, perturbed_edge_index2, perturbed_weight2 = stealthy_adversarial_attack(data, model, perturbation_ratio=0.2)
    output1 = model(perturbed_x1, perturbed_edge_index1, perturbed_weight1)
    output2 = model(perturbed_x2, perturbed_edge_index2, perturbed_weight2)
    loss = contrastive_loss(output1, output2)
    loss.backward()
    optimizer.step()
    return loss.item()

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
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 隐蔽式对抗攻击（支持加权图）
def stealthy_adversarial_attack(data, model, epsilon=0.1, perturbation_ratio=0.3):
    """
    学习CrossBA的多目标优化策略，实现高ASR和隐蔽性的平衡
    """
    # CrossBA参数设置 - 增强攻击强度
    reg_param_1 = 0.2  # 降低隐蔽性权重，提高攻击效果
    reg_param_2 = 0.15  # 降低触发器一致性权重
    
    # 选择更有影响力的目标节点
    target_node = select_high_impact_node(data, model)
    
    # 生成目标嵌入（CrossBA核心）
    target_embedding = generate_target_embedding(data, model, target_node)
    
    # 增加触发器节点数量
    edge_index, edge_weight = add_trigger_nodes(data, target_node, num_trigger_nodes=8)
    edge_index, edge_weight = add_trigger_induced_subgraph(data, edge_index, edge_weight, perturbation_ratio * 1.2)
    edge_index, edge_weight = structure_perturbation(data, edge_index, edge_weight, perturbation_ratio * 1.1)
    
    # 使用CrossBA多目标优化 - 增强攻击参数
    perturbed_x = crossba_perturb_node_features(data.x, model, edge_index, edge_weight, 
                                               perturbation_ratio * 1.3, epsilon * 1.5, target_embedding, 
                                               reg_param_1, reg_param_2)
    
    # 多次迭代攻击以增强效果
    max_iterations = 3
    for i in range(max_iterations):
        # 重新克隆以避免requires_grad错误
        perturbed_x = perturbed_x.clone().detach()
        perturbed_x = crossba_perturb_node_features(perturbed_x, model, edge_index, edge_weight, 
                                                   perturbation_ratio * 0.8, epsilon * 0.7, target_embedding, 
                                                   reg_param_1, reg_param_2)
    
    return perturbed_x, edge_index, edge_weight

# 选择高影响力节点
def select_high_impact_node(data, model):
    """
    选择对模型预测影响最大的节点作为攻击目标
    """
    model.eval()
    with tc.no_grad():
        # 计算每个节点的重要性
        node_importance = []
        for i in range(min(100, data.num_nodes)):  # 限制计算量
            # 计算移除该节点后的损失变化
            original_output = model(data.x, data.edge_index, data.edge_weight)
            original_loss = F.cross_entropy(original_output, tc.arange(data.num_nodes).to(data.x.device) % 7)
            
            # 模拟移除节点i的影响
            masked_x = data.x.clone()
            masked_x[i] = 0  # 将节点i的特征置零
            
            masked_output = model(masked_x, data.edge_index, data.edge_weight)
            masked_loss = F.cross_entropy(masked_output, tc.arange(data.num_nodes).to(data.x.device) % 7)
            
            importance = (original_loss - masked_loss).item()
            node_importance.append(importance)
        
        # 选择重要性最高的节点
        node_importance = tc.tensor(node_importance)
        top_nodes = tc.argsort(node_importance, descending=True)[:10]
        return top_nodes[0].item()

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
        trigger_edge_index = tc.tensor([[0], [0]], dtype=tc.long).to(data.x.device)
        trigger_edge_weight = tc.tensor([1.0], dtype=tc.float).to(data.x.device)
        
        # 获取触发器嵌入
        target_embedding = model(trigger_x, trigger_edge_index, trigger_edge_weight)
        return target_embedding

# CrossBA多目标优化特征扰动
def crossba_perturb_node_features(x, model, edge_index, edge_weight, perturbation_ratio, epsilon, 
                                 target_embedding, reg_param_1, reg_param_2):
    """
    实现CrossBA的多目标优化特征扰动
    """
    model.eval()
    x.requires_grad = True
    output = model(x, edge_index, edge_weight)
    loss = F.cross_entropy(output, tc.arange(x.size(0)).to(x.device) % 7)  # 假设7个类别
    model.zero_grad()
    loss.backward()
    grad = x.grad

    # 计算特征重要性
    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio * 1.5)
    important_features = tc.argsort(feature_importance)[-top_k_features:]

    # 自适应扰动强度 - 增强攻击
    feature_variance = tc.var(x, dim=0)
    adaptive_epsilon = epsilon * feature_variance / (feature_variance.mean() + 1e-8)
    adaptive_epsilon *= 1.2  # 增加扰动强度

    # CrossBA多目标优化
    perturbed_x = x.clone()
    
    # 策略1: 基于梯度的扰动（攻击导向）
    for idx in important_features:
        perturbed_x[:, idx] += adaptive_epsilon[idx] * grad[:, idx].sign()
    
    # 策略2: CrossBA隐蔽性保持
    with tc.no_grad():
        original_emb = model(x, edge_index, edge_weight)
        perturbed_emb = model(perturbed_x, edge_index, edge_weight)
        
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

# 触发节点攻击（节点中毒）
def add_trigger_nodes(data, target_node, num_trigger_nodes=5):
    num_nodes = data.x.size(0)
    trigger_nodes = np.random.choice(num_nodes, num_trigger_nodes, replace=False)
    edge_index = data.edge_index.clone()
    edge_weight = data.edge_weight.clone()
    for node in trigger_nodes:
        new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long)
        edge_index = tc.cat([edge_index, new_edges], dim=1)
        cos_sim = F.cosine_similarity(data.x[target_node].unsqueeze(0), data.x[node].unsqueeze(0)).item()
        edge_weight = tc.cat([edge_weight, tc.tensor([cos_sim, cos_sim], dtype=tc.float).to(edge_weight.device)])
    edge_weight = edge_weight / edge_weight.sum()
    return edge_index, edge_weight

# 触发诱导子图攻击（图中毒）
def add_trigger_induced_subgraph(data, edge_index, edge_weight, perturbation_ratio=0.2):
    num_nodes = data.x.size(0)
    subgraph_size = int(num_nodes * perturbation_ratio)
    perturb_nodes = np.random.choice(num_nodes, subgraph_size, replace=False)
    for node in perturb_nodes:
        for _ in range(5):
            target_node = np.random.choice(num_nodes)
            if node != target_node:
                new_edges = tc.tensor([[node, target_node], [target_node, node]], dtype=tc.long)
                edge_index = tc.cat([edge_index, new_edges], dim=1)
                cos_sim = F.cosine_similarity(data.x[node].unsqueeze(0), data.x[target_node].unsqueeze(0)).item()
                edge_weight = tc.cat([edge_weight, tc.tensor([cos_sim, cos_sim], dtype=tc.float).to(edge_weight.device)])
    edge_weight = edge_weight / edge_weight.sum()
    return edge_index, edge_weight

# 结构扰动：随机添加或删除边
def structure_perturbation(data, edge_index, edge_weight, perturbation_ratio=0.2):
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)
    delete_indices = np.random.choice(num_edges, num_perturbations, replace=False)
    remaining_edges = np.delete(edge_index.numpy(), delete_indices, axis=1)
    remaining_weights = np.delete(edge_weight.numpy(), delete_indices)
    edge_index = tc.tensor(remaining_edges, dtype=tc.long)
    edge_weight = tc.tensor(remaining_weights, dtype=tc.float).to(edge_index.device)
    num_nodes = data.x.size(0)
    new_edges = set()
    while len(new_edges) < num_perturbations:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v:
            new_edges.add((min(u, v), max(u, v)))
    new_edges = np.array(list(new_edges)).T
    edge_index = tc.cat([edge_index, tc.tensor(new_edges, dtype=tc.long)], dim=1)
    for u, v in new_edges.T:
        cos_sim = F.cosine_similarity(data.x[u].unsqueeze(0), data.x[v].unsqueeze(0)).item()
        edge_weight = tc.cat([edge_weight, tc.tensor([cos_sim], dtype=tc.float).to(edge_weight.device)])
    edge_weight = edge_weight / edge_weight.sum()
    return edge_index, edge_weight

# 特征扰动：对节点特征添加噪声
def perturb_node_features(x, edge_index, edge_weight, perturbation_ratio=0.1, epsilon=0.1):
    x = x.clone().detach().requires_grad_(True)
    output = model(x, edge_index, edge_weight)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    model.zero_grad()
    loss.backward()
    grad_sign = x.grad.sign()
    perturbed_x = x + epsilon * grad_sign
    return perturbed_x

# 微调阶段：测试模型
def test():
    model.eval()
    output = model(data.x, data.edge_index, data.edge_weight)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# 执行攻击并微调
def perform_attack_and_finetune():
    perturbed_x, perturbed_edge_index, perturbed_edge_weight = stealthy_adversarial_attack(data, model)
    for epoch in range(100):
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            print(f'微调阶段 Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    return perturbed_x, perturbed_edge_index, perturbed_edge_weight

# 计算置信度
def calculate_confidence(output, mask):
    prob = F.softmax(output[mask], dim=1)
    max_probs = prob.max(dim=1)[0]
    confidence = max_probs.mean().item()
    return confidence

# 设置提示图
prompt_graph = tc.zeros(data.num_nodes, nfeat)
for i in range(data.num_nodes):
    neighbors = data.edge_index[1][data.edge_index[0] == i]
    if len(neighbors) > 0:
        neighbor_weights = data.edge_weight[data.edge_index[0] == i]
        neighbor_features = data.x[neighbors]
        prompt_graph[i] = (neighbor_features * neighbor_weights.unsqueeze(1)).sum(dim=0) / neighbor_weights.sum()
model.set_prompt_graph(prompt_graph)

def evaluate_with_metrics(perturbed_x, perturbed_edge_index, perturbed_edge_weight):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index, perturbed_edge_weight)
    y_pred_pos = output[data.test_mask].cpu().detach().numpy().flatten()
    y_pred_neg = output[~data.test_mask].cpu().detach().numpy()
    if len(y_pred_pos) != y_pred_neg.shape[0]:
        print(f"警告: 正样本数量 ({len(y_pred_pos)}) 和负样本数量 ({y_pred_neg.shape[0]}) 不一致，将进行随机下采样对齐。")
        min_samples = min(len(y_pred_pos), y_pred_neg.shape[0])
        y_pred_pos = np.random.choice(y_pred_pos, min_samples, replace=False)
        y_pred_neg = y_pred_neg[:min_samples]
    evaluator_hits = Evaluator(eval_metric='hits@50')
    evaluator_mrr = Evaluator(eval_metric='mrr')
    hits_at_k = evaluator_hits.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    mrr_score = evaluator_mrr.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    print(f"Hits@50: {hits_at_k['hits@50']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    return hits_at_k, mrr_score

# 启动训练过程
for epoch in range(100):
    loss = train_with_contrastive_loss()
    if epoch % 10 == 0:
        print(f'预训练阶段 Epoch: {epoch:03d}, Loss: {loss:.4f}')

perturbed_x, perturbed_edge_index, perturbed_edge_weight = perform_attack_and_finetune()

# 测试攻击后的模型性能
def test_after_attack(perturbed_x, perturbed_edge_index, perturbed_edge_weight):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index, perturbed_edge_weight)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# 计算攻击成功率
def calculate_asr(original_pred, perturbed_pred, test_mask, y_true):
    # 修复ASR计算：比较原始预测和攻击后预测的差异
    successful_attack = (perturbed_pred[test_mask] != original_pred[test_mask]).sum().item()
    total_attack = test_mask.sum().item()
    asr = successful_attack / total_attack if total_attack > 0 else 0
    return asr

# 计算扰动比例
def calculate_feature_perturbation_ratio(original_x, perturbed_x):
    perturbation = (original_x - perturbed_x).abs()
    feature_perturbation_ratio = perturbation.mean().item()
    return feature_perturbation_ratio

# 计算攻击后模型的各项指标
def evaluate_attack_metrics():
    original_pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
    perturbed_pred = model(perturbed_x, perturbed_edge_index, perturbed_edge_weight).argmax(dim=1)
    asr = calculate_asr(original_pred, perturbed_pred, data.test_mask, data.y)
    print(f'攻击成功率 (ASR): {asr:.4f}')
    feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
    print(f'扰动比例: {feature_perturbation_ratio:.4f}')
    original_confidence = calculate_confidence(model(data.x, data.edge_index, data.edge_weight), data.test_mask)
    perturbed_confidence = calculate_confidence(model(perturbed_x, perturbed_edge_index, perturbed_edge_weight), data.test_mask)
    confidence_drop = original_confidence - perturbed_confidence
    print(f'原始置信度: {original_confidence:.4f}')
    print(f'攻击后置信度: {perturbed_confidence:.4f}')
    print(f'置信度下降: {confidence_drop:.4f}')
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = perturbed_pred[data.test_mask].cpu().numpy()
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'精确度: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1 分数: {f1:.4f}')
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(i) for i in range(dataset.num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis_r', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('Cora - 混淆矩阵')
    plt.show()
    y_pred_probs = F.softmax(model(perturbed_x, perturbed_edge_index, perturbed_edge_weight)[data.test_mask], dim=1).detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    print(f'ROC-AUC: {auc:.4f}')
    specificity_per_class = []
    for i in range(dataset.num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)
    print(f'特异度: {specificity:.4f}')

original_acc = test()
print(f'原始准确率 (攻击前): {original_acc:.4f}')
perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index, perturbed_edge_weight)
print(f'攻击后准确率: {perturbed_acc:.4f}')
evaluate_attack_metrics()
evaluate_with_metrics(perturbed_x, perturbed_edge_index, perturbed_edge_weight)

# 资源使用评估
def evaluate_resource_usage():
    gpu_mem_usage = []
    cpu_mem_usage = []
    time_delays = []
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
    print("\n评估原始模型资源使用...")
    start_time = time.time()
    with tc.no_grad():
        model.eval()
        original_output = model(data.x, data.edge_index, data.edge_weight)
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
        perturbed_output = model(perturbed_x, perturbed_edge_index, perturbed_edge_weight)
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
def profile_computation():
    print("\n运行计算性能分析...")
    with tc.profiler.profile(
            activities=[tc.profiler.ProfilerActivity.CPU, tc.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_flops=True
    ) as prof:
        with tc.no_grad():
            model(data.x, data.edge_index, data.edge_weight)
        with tc.no_grad():
            model(perturbed_x, perturbed_edge_index, perturbed_edge_weight)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("gcn_performance_trace.json")
    print("性能分析跟踪已保存到 gcn_performance_trace.json")

evaluate_resource_usage()
if tc.cuda.is_available():
    profile_computation()
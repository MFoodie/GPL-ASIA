import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import random
import os
import time
import psutil
import torch.cuda as cuda

# 设备设置
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if tc.cuda.is_available():
    print(f"GPU名称: {tc.cuda.get_device_name(0)}")
    print(f"GPU内存: {tc.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    tc.cuda.empty_cache()
    tc.backends.cudnn.benchmark = True

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

# 将数据移动到GPU
data = data.to(device)
print(f"数据已移动到 {device}")
print(f"节点特征形状: {data.x.shape}")
print(f"边索引形状: {data.edge_index.shape}")

has_printed = False

# Evaluator 类
afterward_print = False
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

# 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.prompt_graph = None
    def forward(self, x, edge_index):
        if self.prompt_graph is not None:
            x = x + self.prompt_graph
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    def set_prompt_graph(self, prompt_graph):
        self.prompt_graph = prompt_graph
        global has_printed
        if not has_printed:
            print(f'ProG graph set with shape: {prompt_graph.shape}')
            has_printed = True

# 定义优化器和损失函数
nfeat = dataset.num_features
nhid = 8
nclass = dataset.num_classes
dropout = 0.6

model = GCN(nfeat, nclass, nhid, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 生成 ProG 提示图
def generate_prog_graph(data):
    prog_graph = tc.randn_like(data.x) * 0.1
    return prog_graph

# 微调阶段：标准训练
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    loss.backward()
    optimizer.step()
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
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

def train_with_contrastive_loss_simple():
    """
    增强的对比学习预训练，使用多样化数据增强提高攻击成功率，保持纯黑盒方法
    """
    model.train()
    optimizer.zero_grad()
    
    # 使用多样化的数据增强替代复杂攻击
    # 增强1：特征噪声扰动
    noise_scale1 = 0.1 + tc.rand(1).item() * 0.1  # 0.1-0.2的随机噪声
    x1 = data.x + tc.randn_like(data.x) * noise_scale1
    
    # 增强2：特征掩码和缩放
    mask_ratio = 0.1 + tc.rand(1).item() * 0.1  # 10-20%的特征掩码
    mask = tc.rand_like(data.x) > mask_ratio
    x2 = data.x * mask + tc.randn_like(data.x) * 0.05
    
    # 增强3：特征缩放扰动
    scale_factor = 0.9 + tc.rand(1).item() * 0.2  # 0.9-1.1的缩放
    x1 = x1 * scale_factor
    x2 = x2 * (2.0 - scale_factor)  # 互补缩放
    
    e1 = data.edge_index
    e2 = data.edge_index
    
    prog_graph = generate_prog_graph(data)
    model.set_prompt_graph(prog_graph)
    output1 = model(x1, e1)
    output2 = model(x2, e2)
    loss = contrastive_loss(output1, output2)
    loss.backward()
    optimizer.step()
    return loss.item()

def optimize_attack_independently():
    """
    独立的攻击优化阶段，参考CrossBA的10个epoch优化
    """
    print("优化攻击策略...")
    best_attack = None
    best_asr = 0
    
    for epoch in range(15):  # 增加到15个epoch提高攻击成功率
        print(f"攻击优化 Epoch {epoch+1}/15")
        
        # 执行攻击
        perturbed_x, perturbed_edge_index = stealthy_adversarial_attack(data, model)
        
        # 评估攻击效果
        # 计算真正的ASR：基于预测变化
        model.eval()
        with tc.no_grad():
            original_pred = model(data.x, data.edge_index).argmax(dim=1)
            perturbed_pred = model(perturbed_x, perturbed_edge_index).argmax(dim=1)
            asr = calculate_asr(original_pred, perturbed_pred, data.test_mask.to(device), data.y.to(device))
            
            # 计算准确率
            test_mask = data.test_mask.to(device)
            original_acc = (original_pred[test_mask] == data.y[test_mask].to(device)).float().mean().item()
            perturbed_acc = (perturbed_pred[test_mask] == data.y[test_mask].to(device)).float().mean().item()
        
        print(f"  Original Acc: {original_acc:.4f}, Perturbed Acc: {perturbed_acc:.4f}, ASR: {asr:.4f}")
        
        # 保存最佳攻击
        if asr > best_asr:
            best_asr = asr
            best_attack = (perturbed_x, perturbed_edge_index)
    
    print(f"最佳攻击ASR: {best_asr:.4f}")
    return best_attack if best_attack else stealthy_adversarial_attack(data, model)

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
def stealthy_adversarial_attack(data, model, epsilon=0.25, perturbation_ratio=0.6, task_type='node_classification'):
    original_acc = test()
    perturbation_ratio = min(perturbation_ratio, 0.8 * (1 - original_acc) if task_type == 'node_classification' else 0.6)
    target_nodes = select_multiple_target_nodes(data, task_type, num_targets=5)
    perturbed_edge_index = add_enhanced_trigger_nodes(data, target_nodes, num_trigger_nodes=10 if task_type == 'node_classification' else 7)
    perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
    perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)
    perturbed_x = enhanced_perturb_node_features(data.x, perturbation_ratio, epsilon, task_type)
    max_attempts = 6
    attempt = 0
    perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
    target_acc_reduction = 0.20
    while perturbed_acc > original_acc * (1 - target_acc_reduction) and attempt < max_attempts:
        perturbation_ratio = min(perturbation_ratio * 1.6, 0.9)
        epsilon = min(epsilon * 1.5, 0.5)
        perturbed_edge_index = add_enhanced_trigger_nodes(data, target_nodes, num_trigger_nodes=15 if task_type == 'node_classification' else 10)
        perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type)
        perturbed_edge_index = structure_perturbation(data, perturbed_edge_index, perturbation_ratio)
        perturbed_x = enhanced_perturb_node_features(data.x, perturbation_ratio, epsilon, task_type)
        perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
        attempt += 1
        print(f"Attack Attempt {attempt}: Perturbed Acc = {perturbed_acc:.4f}, Target: {original_acc * (1 - target_acc_reduction):.4f}, Perturbation Ratio = {perturbation_ratio:.4f}")
    return perturbed_x, perturbed_edge_index

# 攻击成功率统计
def calculate_attack_success_rate(original_pred, perturbed_pred, test_mask, y_true):
    mask = test_mask.to(device)
    successful_attack = (perturbed_pred[mask] != original_pred[mask]).to(tc.long).sum().item()
    total_attack = mask.to(tc.long).sum().item()
    asr = successful_attack / max(1, total_attack)
    class_asr = {}
    for class_id in range(y_true.max().item() + 1):
        class_mask = (y_true[mask] == class_id)
        class_total = class_mask.to(tc.long).sum().item()
        if class_total > 0:
            class_successful = (perturbed_pred[mask][class_mask] != original_pred[mask][class_mask]).to(tc.long).sum().item()
            class_asr[class_id] = class_successful / class_total
    return asr, class_asr

# 选择多个目标节点
def select_multiple_target_nodes(data, task_type='node_classification', num_targets=5):
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    target_nodes = []
    if task_type == 'node_classification':
        test_nodes = tc.where(data.test_mask.to(device))[0]
        if len(test_nodes) > 0:
            test_degrees = degrees[test_nodes]
            for class_id in range(data.y.to(device).max().item() + 1):
                class_nodes = test_nodes[data.y.to(device)[test_nodes] == class_id]
                if len(class_nodes) > 0:
                    class_degrees = degrees[class_nodes]
                    low_degree_class_nodes = class_nodes[tc.argsort(class_degrees)[:3]]
                    target_nodes.extend(low_degree_class_nodes.cpu().numpy().tolist())
        if len(target_nodes) < num_targets:
            low_degree_nodes = tc.argsort(degrees)[:30]
            additional_nodes = low_degree_nodes.cpu().numpy().tolist()
            target_nodes.extend(additional_nodes[:num_targets - len(target_nodes)])
    else:
        mid_degree_nodes = tc.argsort(degrees)[int(0.2 * len(degrees)):int(0.6 * len(degrees))]
        target_nodes = mid_degree_nodes.cpu().numpy().tolist()[:num_targets]
    return target_nodes[:num_targets]

# 增强触发节点攻击
def add_enhanced_trigger_nodes(data, target_nodes, num_trigger_nodes=10):
    num_nodes = data.x.size(0)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    low_degree_nodes = tc.argsort(degrees)[:num_trigger_nodes * 4]
    selected_nodes = np.random.choice(low_degree_nodes.cpu().numpy(), num_trigger_nodes, replace=False)
    edge_index = data.edge_index.clone()
    for target_node in target_nodes:
        for node in selected_nodes:
            new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    for i in range(len(selected_nodes) - 1):
        for j in range(i + 1, len(selected_nodes)):
            trigger_edge = tc.tensor([[selected_nodes[i], selected_nodes[j]],
                                    [selected_nodes[j], selected_nodes[i]]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, trigger_edge], dim=1).to(device)
    return edge_index

# 增强特征扰动 - 基于CrossBA的纯黑盒攻击
def enhanced_perturb_node_features(x, perturbation_ratio=0.4, epsilon=0.25, task_type='node_classification', attack_strategy='adaptive'):
    """
    基于CrossBA方法的纯黑盒特征扰动，不依赖梯度信息
    """
    perturbed_x = x.clone()
    
    # 计算特征重要性（基于统计特征，不依赖梯度）
    feature_variance = tc.var(x, dim=0)
    feature_mean = tc.mean(x, dim=0)
    feature_importance = feature_variance / (feature_mean.abs() + 1e-8)
    
    # 选择重要特征进行扰动
    top_k_features = int(x.size(1) * perturbation_ratio * 2.5)
    important_features = tc.argsort(feature_importance)[-top_k_features:]
    
    # 自适应扰动强度
    adaptive_epsilon = epsilon * feature_variance / (feature_variance.mean() + 1e-8)
    if task_type == 'link_prediction':
        adaptive_epsilon *= 0.8
    else:
        adaptive_epsilon *= 1.5
    
    if attack_strategy == 'adaptive':
        # 自适应攻击：根据特征分布调整扰动
        for idx in important_features:
            # 基于特征方差的扰动
            perturbation = tc.randn_like(x[:, idx]) * adaptive_epsilon[idx] * (1 + feature_variance[idx].item())
            perturbed_x[:, idx] += perturbation
            
    elif attack_strategy == 'degree_based':
        # 基于度数的攻击（需要传入度数信息）
        degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        for idx in important_features:
            # 对高度节点施加更大扰动
            degree_factor = 1.0 + tc.mean(degrees.float()) / tc.max(degrees.float())
            perturbation = tc.randn_like(x[:, idx]) * adaptive_epsilon[idx] * degree_factor
            perturbed_x[:, idx] += perturbation
            
    else:  # random
        # 随机攻击
        for idx in important_features:
            perturbation = tc.randn_like(x[:, idx]) * adaptive_epsilon[idx]
            perturbed_x[:, idx] += perturbation
    
    # 添加噪声增强隐蔽性
    noise_ratio = 0.4
    noise_features = important_features[:int(len(important_features) * noise_ratio)]
    for idx in noise_features:
        noise = tc.randn_like(perturbed_x[:, idx]) * adaptive_epsilon[idx] * 0.8
        perturbed_x[:, idx] += noise
    
    # 针对测试节点的特殊处理
    if task_type == 'node_classification':
        test_mask = data.test_mask.to(device)
        if test_mask.sum() > 0:
            # 对测试节点施加更强的扰动
            for idx in important_features:
                test_perturbation = tc.randn_like(perturbed_x[test_mask, idx]) * adaptive_epsilon[idx] * 2.0
                perturbed_x[test_mask, idx] += test_perturbation
    
    # 限制扰动范围
    perturbed_x = tc.clamp(perturbed_x, min=x.min() - 0.2, max=x.max() + 0.2)
    
    return perturbed_x

# 其他辅助函数（select_low_centrality_node/add_trigger_nodes/add_trigger_induced_subgraph/structure_perturbation/perturb_node_features）

def select_low_centrality_node(data, task_type='node_classification'):
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    if task_type == 'node_classification':
        test_nodes = tc.where(data.test_mask.to(device))[0]
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
        new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long).to(device)
        edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index

def add_trigger_induced_subgraph(data, perturbation_ratio=0.2, task_type='node_classification'):
    num_nodes = data.x.size(0)
    edge_density = data.edge_index.size(1) / (num_nodes * (num_nodes - 1))
    perturbation_ratio = min(perturbation_ratio, 0.1 / edge_density if task_type == 'node_classification' else 0.5 / edge_density)
    subgraph_size = int(num_nodes * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    test_nodes = tc.where(data.test_mask.to(device))[0]
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
            new_edges = tc.tensor([[node, target_node], [target_node, node]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index

def structure_perturbation(data, edge_index, perturbation_ratio=0.4):
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    edge_importance = degrees[edge_index[0]] + degrees[edge_index[1]]
    if hasattr(data, 'x') and data.x is not None:
        src_features = data.x[edge_index[0]]
        dst_features = data.x[edge_index[1]]
        feature_similarity = F.cosine_similarity(src_features, dst_features, dim=1)
        edge_importance = edge_importance + feature_similarity * 15
    low_importance_edges = tc.argsort(edge_importance)[:int(num_perturbations * 1.5)]
    remaining_edges = np.delete(edge_index.cpu().numpy(), low_importance_edges.cpu().numpy(), axis=1)
    remaining_edges = tc.tensor(remaining_edges, dtype=tc.long).to(device)
    new_edges = set()
    test_nodes = tc.where(data.test_mask.to(device))[0].cpu().numpy()
    low_degree_nodes = tc.argsort(degrees)[:int(data.num_nodes * 0.5)].cpu().numpy()
    for _ in range(int(num_perturbations * 0.7)):
        if len(test_nodes) > 0:
            u = np.random.choice(test_nodes)
            v = np.random.choice(low_degree_nodes)
            if u != v:
                new_edges.add((min(u, v), max(u, v)))
    for _ in range(int(num_perturbations * 0.3)):
        u, v = np.random.choice(low_degree_nodes, 2, replace=False)
        new_edges.add((min(u, v), max(u, v)))
    while len(new_edges) < num_perturbations:
        u = np.random.choice(test_nodes if len(test_nodes) > 0 else low_degree_nodes)
        v = np.random.choice(low_degree_nodes)
        if u != v:
            new_edges.add((min(u, v), max(u, v)))
    new_edges = np.array(list(new_edges)).T
    perturbed_edge_index = tc.cat([remaining_edges, tc.tensor(new_edges, dtype=tc.long).to(device)], dim=1)
    return perturbed_edge_index

def perturb_node_features(x, perturbation_ratio=0.2, epsilon=0.15, task_type='node_classification'):
    data.x.requires_grad = True
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask.to(device)], data.y.to(device)[data.train_mask.to(device)])
    model.zero_grad()
    loss.backward()
    grad = data.x.grad
    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio * 1.5)
    important_features = tc.argsort(feature_importance)[-top_k_features:]
    feature_variance = tc.var(data.x, dim=0)
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
    mask = data.test_mask.to(device)
    correct = (pred[mask] == data.y.to(device)[mask]).sum().item()
    acc = correct / float(mask.sum().item())
    return acc

# 执行攻击并微调
def perform_attack_and_finetune():
    print("开始微调...")
    # 使用之前优化的最佳攻击
    perturbed_x, perturbed_edge_index = stealthy_adversarial_attack(data, model)
    
    # 微调模型（参考CrossBA的20-30个epoch）
    for epoch in range(30):  # 减少到30个epoch，参考CrossBA
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            if tc.cuda.is_available():
                gpu_mem = tc.cuda.memory_allocated() / 1024**2
                print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, GPU内存: {gpu_mem:.1f}MB')
            else:
                print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    return perturbed_x, perturbed_edge_index

# 对抗扰动损失（未直接使用）
def adversarial_loss(x, edge_index, target, model, epsilon=0.1):
    x.requires_grad = True
    output = model(x, edge_index)
    loss = criterion(output[data.train_mask.to(device)], target[data.train_mask.to(device)])
    model.zero_grad()
    loss.backward()
    grad_x = x.grad
    perturbation = epsilon * grad_x.sign()
    return perturbation, loss.item()

# 置信度
def calculate_confidence(output, mask):
    prob = F.softmax(output[mask], dim=1)
    max_probs = prob.max(dim=1)[0]
    confidence = max_probs.mean().item()
    return confidence

# 设置ProG
prog_graph = tc.randn(data.num_nodes, nfeat).to(device)
model.set_prompt_graph(prog_graph)

# 修复 evaluate_with_metrics
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

# 攻击后测试
def test_after_attack(perturbed_x, perturbed_edge_index):
    model.eval()
    output = model(perturbed_x, perturbed_edge_index)
    pred = output.argmax(dim=1)
    mask = data.test_mask.to(device)
    correct = (pred[mask] == data.y.to(device)[mask]).sum().item()
    acc = correct / float(mask.sum().item())
    return acc

# ASR
def calculate_asr(original_pred, perturbed_pred, test_mask, y_true):
    mask = test_mask.to(device)
    successful_attack = (perturbed_pred[mask] != original_pred[mask]).to(tc.long).sum().item()
    total_attack = mask.to(tc.long).sum().item()
    asr = successful_attack / max(1, total_attack)
    return asr

# 特征扰动比例
def calculate_feature_perturbation_ratio(original_x, perturbed_x):
    perturbation = (original_x - perturbed_x).abs()
    feature_perturbation_ratio = perturbation.mean().item()
    return feature_perturbation_ratio

# 指标评估
def evaluate_attack_metrics():
    original_pred = model(data.x, data.edge_index).argmax(dim=1)
    perturbed_pred = model(perturbed_x, perturbed_edge_index).argmax(dim=1)
    asr, class_asr = calculate_attack_success_rate(original_pred, perturbed_pred, data.test_mask.to(device), data.y.to(device))
    print(f'Attack Success Rate (ASR): {asr:.4f}')
    print(f'Class-wise ASR: {class_asr}')
    feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
    print(f'Perturbation Ratio: {feature_perturbation_ratio:.4f}')
    original_confidence = calculate_confidence(model(data.x, data.edge_index), data.test_mask.to(device))
    perturbed_confidence = calculate_confidence(model(perturbed_x, perturbed_edge_index), data.test_mask.to(device))
    confidence_drop = original_confidence - perturbed_confidence
    print(f'Original Confidence: {original_confidence:.4f}')
    print(f'Perturbed Confidence: {perturbed_confidence:.4f}')
    print(f'Confidence Drop: {confidence_drop:.4f}')
    y_true = data.y.to(device)[data.test_mask.to(device)].cpu().numpy()
    y_pred = perturbed_pred[data.test_mask.to(device)].cpu().numpy()
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(i) for i in range(dataset.num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis_r', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Cora - Confusion Matrix')
    plt.show()
    y_pred_probs = F.softmax(model(perturbed_x, perturbed_edge_index)[data.test_mask.to(device)], dim=1).detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    print(f'ROC-AUC: {auc:.4f}')
    specificity_per_class = []
    for i in range(dataset.num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)
    print(f'Specificity: {specificity:.4f}')

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
    if tc.cuda.is_available():
        gpu_mem = tc.cuda.memory_allocated() / 1024**2
        print(f"原始模型推理时间: {original_time:.4f}s, GPU内存: {gpu_mem:.1f}MB")
    else:
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
            model(data.x, data.edge_index)
            model(perturbed_x, perturbed_edge_index)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("gcn_performance_trace.json")
    print("性能分析跟踪已保存到 gcn_performance_trace.json")

# 启动训练过程
# 第一阶段：纯对比学习预训练（不执行攻击，参考CrossBA的400 epochs）
for epoch in range(80):  # 增加到80个epoch提高攻击成功率
    loss = train_with_contrastive_loss_simple()  # 使用简化版本
    if epoch % 10 == 0:
        if tc.cuda.is_available():
            gpu_mem = tc.cuda.memory_allocated() / 1024**2
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}, GPU内存: {gpu_mem:.1f}MB')
        else:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("开始攻击优化...")
# 第二阶段：独立的攻击优化（参考CrossBA的10个epoch）
perturbed_x, perturbed_edge_index = optimize_attack_independently()

print("开始微调...")
# 第三阶段：微调（参考CrossBA的20-30个epoch）
perturbed_x, perturbed_edge_index = perform_attack_and_finetune()
original_acc = test()
print(f'Original Accuracy (Before Attack): {original_acc:.4f}')
perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index)
print(f'Perturbed Accuracy (After Attack): {perturbed_acc:.4f}')
evaluate_attack_metrics()
evaluate_with_metrics(perturbed_x, perturbed_edge_index)
evaluate_resource_usage()
if tc.cuda.is_available():
    profile_computation()

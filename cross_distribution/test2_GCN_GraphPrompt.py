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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，确保实验可重复性
seed = 4322
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

# 定义子分布划分标准（基于节点度）
degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
degree_thresholds = np.percentile(degrees, [33, 66])
sub_distributions = {
    'low_degree': tc.tensor(degrees <= degree_thresholds[0], dtype=bool).to(device),
    'mid_degree': tc.tensor((degrees > degree_thresholds[0]) & (degrees <= degree_thresholds[1]), dtype=bool).to(
        device),
    'high_degree': tc.tensor(degrees > degree_thresholds[1], dtype=bool).to(device)
}

# 验证子分布的有效性
for name, mask in sub_distributions.items():
    num_nodes = mask.sum().item()
    print(f"子分布 {name} 包含节点数: {num_nodes}")
    if num_nodes == 0:
        print(f"错误: 子分布 {name} 为空，跳过该子分布！")
        continue
results = {}


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
            raise ValueError(f"未知指标: {self.eval_metric}")


# 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.prompt_graph = None
        self.prompt_optimizer = None

    def forward(self, x, edge_index):
        if self.prompt_graph is not None:
            x = x + self.prompt_graph
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def set_prompt_graph(self, prompt_graph=None):
        if prompt_graph is None:
            prompt_graph = tc.mean(data.x, dim=0).repeat(data.num_nodes, 1)
            prompt_graph += tc.randn_like(prompt_graph) * 0.1
        self.prompt_graph = nn.Parameter(prompt_graph.to(device), requires_grad=True)
        self.prompt_optimizer = optim.Adam([self.prompt_graph], lr=0.01)
        print(f'提示图已设置，形状: {prompt_graph.shape}')


# 定义优化器和损失函数
nfeat = dataset.num_features
nhid = 16
nclass = dataset.num_classes
dropout = 0.5
model = GCN(nfeat, nclass, nhid, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


# 预训练阶段：对比损失
def train_with_contrastive_loss(mask=None):
    model.train()
    optimizer.zero_grad()
    if model.prompt_graph is not None:
        model.prompt_optimizer.zero_grad()
    perturbed_x1, perturbed_edge_index1 = enhanced_stealthy_adversarial_attack(data, model, perturbation_ratio=0.3,
                                                                               mask=mask)
    perturbed_x2, perturbed_edge_index2 = enhanced_stealthy_adversarial_attack(data, model, perturbation_ratio=0.3,
                                                                               mask=mask)
    output1 = model(perturbed_x1 + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index1)
    output2 = model(perturbed_x2 + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index2)
    loss = contrastive_loss(output1[mask] if mask is not None else output1,
                            output2[mask] if mask is not None else output2)
    loss.backward()
    optimizer.step()
    if model.prompt_graph is not None:
        model.prompt_optimizer.step()
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


# 微调阶段：标准训练
def train(mask=None):
    model.train()
    optimizer.zero_grad()
    if model.prompt_graph is not None:
        model.prompt_optimizer.zero_grad()
    output = model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index)
    loss = criterion(
        output[mask & data.train_mask.to(device)] if mask is not None else output[data.train_mask.to(device)],
        data.y.to(device)[mask & data.train_mask.to(device)] if mask is not None else data.y.to(device)[
            data.train_mask.to(device)])
    loss.backward()
    optimizer.step()
    if model.prompt_graph is not None:
        model.prompt_optimizer.step()
    return loss.item()


# 增强型对抗攻击
# CrossBA参数
reg_param_1 = 0.2  # 隐蔽性权重
reg_param_2 = 0.15  # 触发器一致性权重


# 生成目标嵌入
def generate_target_embedding(data, model, target_node):
    """生成目标嵌入用于CrossBA攻击"""
    # 创建触发器节点
    trigger_x = data.x[target_node:target_node + 1].clone()
    trigger_edge_index = tc.tensor([[0], [0]], dtype=tc.long).to(data.x.device)

    # 获取目标嵌入
    with tc.no_grad():
        target_embedding = model(trigger_x, trigger_edge_index)

    return target_embedding


# CrossBA多目标特征扰动
def crossba_perturb_node_features(x, model, edge_index, perturbation_ratio, epsilon, target_embedding, reg_param_1,
                                  reg_param_2, data):
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


def enhanced_stealthy_adversarial_attack(data, model, epsilon=0.2, perturbation_ratio=0.3,
                                         task_type='node_classification', mask=None):
    model.eval()
    num_nodes = data.x.size(0) if mask is None else mask.sum().item()
    original_acc = test(mask)
    perturbation_ratio = min(perturbation_ratio, 0.6 * (1 - original_acc))
    epsilon = min(epsilon, 0.35)

    target_node = select_influential_node(data, model, task_type, mask)
    perturbed_edge_index = add_trigger_nodes(data, target_node, num_trigger_nodes=8, mask=mask)
    perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio * 1.2, task_type, mask)
    perturbed_edge_index = enhanced_structure_perturbation(data, perturbed_edge_index, perturbation_ratio, mask)

    # 生成目标嵌入用于CrossBA攻击
    target_embedding = generate_target_embedding(data, model, target_node)

    # 使用CrossBA多目标优化进行特征扰动
    perturbed_x = crossba_perturb_node_features(data.x, model, data.edge_index,
                                                perturbation_ratio, epsilon, target_embedding,
                                                reg_param_1, reg_param_2, data)

    max_attempts = 5
    attempt = 0
    perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index, mask)
    while perturbed_acc > original_acc * 0.8 and attempt < max_attempts:
        perturbation_ratio = min(perturbation_ratio * 1.5, 0.7)
        epsilon = min(epsilon * 1.3, 0.4)
        perturbed_edge_index = add_trigger_induced_subgraph(data, perturbation_ratio, task_type, mask)
        perturbed_edge_index = enhanced_structure_perturbation(data, perturbed_edge_index, perturbation_ratio, mask)

        # 重新生成目标嵌入并应用CrossBA攻击
        target_embedding = generate_target_embedding(data, model, target_node)
        perturbed_x = crossba_perturb_node_features(data.x, model, data.edge_index,
                                                    perturbation_ratio, epsilon, target_embedding,
                                                    reg_param_1, reg_param_2, data)
        perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index, mask)
        attempt += 1
        print(f"攻击尝试 {attempt}: 扰动后准确率 = {perturbed_acc:.4f}, 扰动比例 = {perturbation_ratio:.4f}")

    return perturbed_x, perturbed_edge_index


def select_influential_node(data, model, task_type='node_classification', mask=None):
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    if mask is not None:
        masked_degrees = degrees[mask]
        global_test_mask = data.test_mask.to(device) & mask
        test_nodes = tc.where(global_test_mask)[0].to(device)
        if len(test_nodes) > 0:
            mask_indices = tc.nonzero(mask)[0]
            test_mask_indices = tc.searchsorted(mask_indices, test_nodes)
            valid_mask = test_mask_indices < len(masked_degrees)
            test_degrees = masked_degrees[test_mask_indices[valid_mask]]
            test_nodes = test_nodes[valid_mask]
            test_uncertainty_indices = test_mask_indices[valid_mask]
        else:
            test_degrees = tc.tensor([], dtype=degrees.dtype).to(device)
            test_uncertainty_indices = tc.tensor([], dtype=tc.long).to(device)
    else:
        test_nodes = tc.where(data.test_mask.to(device))[0].to(device)
        test_degrees = degrees[test_nodes]
        test_uncertainty_indices = test_nodes

    with tc.no_grad():
        output = model(data.x, data.edge_index)
        probs = F.softmax(output, dim=1)
        uncertainty = -(probs * tc.log(probs + 1e-10)).sum(dim=1)
        if mask is not None:
            uncertainty = uncertainty[mask]

    if len(test_nodes) > 0:
        test_uncertainty = uncertainty[test_uncertainty_indices]
        score = test_uncertainty + 0.5 * test_degrees / test_degrees.max()
        influential_nodes = test_nodes[tc.argsort(score, descending=True)[:10]]
    else:
        score = uncertainty + 0.5 * degrees / degrees.max()
        influential_nodes = tc.argsort(score, descending=True)[:10]

    class_counts = tc.bincount(data.y.to(device)[data.train_mask.to(device) & (
        mask if mask is not None else tc.ones_like(data.train_mask.to(device), dtype=bool))])
    minority_class = tc.argmin(class_counts).item()
    minority_nodes = influential_nodes[data.y.to(device)[influential_nodes] == minority_class]
    if len(minority_nodes) > 0:
        return minority_nodes[np.random.randint(0, len(minority_nodes))].item()
    return influential_nodes[0].item()


def add_trigger_nodes(data, target_node, num_trigger_nodes=8, mask=None):
    num_nodes = data.x.size(0)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    low_degree_nodes = tc.argsort(degrees)[:num_trigger_nodes * 2]
    if mask is not None:
        low_degree_nodes = low_degree_nodes[mask[low_degree_nodes]]
    if len(low_degree_nodes) == 0:
        print(f"警告: 子分布 {mask is not None and 'Masked' or 'Full'} 的 low_degree_nodes 为空，使用未掩码的低度节点")
        low_degree_nodes = tc.argsort(degrees)[:num_trigger_nodes * 2]
    if len(low_degree_nodes) < num_trigger_nodes:
        print(f"警告: low_degree_nodes 数量不足 ({len(low_degree_nodes)} < {num_trigger_nodes})，跳过触发节点添加")
        return data.edge_index
    selected_nodes = np.random.choice(low_degree_nodes.cpu().numpy(), num_trigger_nodes, replace=False)
    edge_index = data.edge_index.clone().to(device)
    for node in selected_nodes:
        new_edges = tc.tensor([[target_node, node], [node, target_node]], dtype=tc.long).to(device)
        edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index


def add_trigger_induced_subgraph(data, perturbation_ratio=0.3, task_type='node_classification', mask=None):
    num_nodes = data.x.size(0)
    edge_density = data.edge_index.size(1) / (num_nodes * (num_nodes - 1))
    perturbation_ratio = min(perturbation_ratio, 0.15 / edge_density)
    subgraph_size = int(num_nodes * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    test_nodes = tc.where(data.test_mask.to(device) & (
        mask if mask is not None else tc.ones_like(data.test_mask.to(device), dtype=bool).to(device)))[0]
    if len(test_nodes) > 0:
        test_degrees = degrees[test_nodes]
        perturb_nodes = test_nodes[tc.argsort(test_degrees)[:subgraph_size]].cpu().numpy()
    else:
        if num_nodes < subgraph_size:
            print(f"警告: 节点总数 ({num_nodes}) 小于子图大小 ({subgraph_size})，跳过子图添加")
            return data.edge_index
        perturb_nodes = np.random.choice(num_nodes, subgraph_size, replace=False)
    if mask is not None:
        perturb_nodes = perturb_nodes[mask[perturb_nodes].cpu().numpy()]
    if len(perturb_nodes) == 0:
        print(f"警告: 子分布 {mask is not None and 'Masked' or 'Full'} 的 perturb_nodes 为空，跳过子图添加")
        return data.edge_index

    edge_index = data.edge_index.clone().to(device)
    low_degree_nodes = tc.argsort(degrees)[:int(num_nodes * 0.3)].cpu().numpy()
    if len(low_degree_nodes) == 0:
        print(f"警告: low_degree_nodes 为空，跳过子图添加")
        return edge_index
    num_edges = 4 if task_type == 'node_classification' else 3
    for node in perturb_nodes:
        for _ in range(num_edges):
            target_node = np.random.choice(low_degree_nodes)
            new_edges = tc.tensor([[node, target_node], [target_node, node]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1).to(device)
    return edge_index


def enhanced_structure_perturbation(data, edge_index, perturbation_ratio=0.3, mask=None):
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)
    degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    edge_importance = degrees[edge_index[0]] + degrees[edge_index[1]]
    high_importance_edges = tc.argsort(edge_importance, descending=True)[:num_perturbations]
    remaining_edges = np.delete(edge_index.cpu().numpy(), high_importance_edges.cpu().numpy(), axis=1)
    remaining_edges = tc.tensor(remaining_edges, dtype=tc.long).to(device)

    new_edges = set()
    test_nodes = tc.where(data.test_mask.to(device) & (
        mask if mask is not None else tc.ones_like(data.test_mask.to(device), dtype=bool).to(device)))[0].cpu().numpy()
    low_degree_nodes = tc.argsort(degrees)[:int(data.num_nodes * 0.3)].cpu().numpy()
    if len(low_degree_nodes) == 0:
        print(f"警告: low_degree_nodes 为空，跳过结构扰动")
        return edge_index
    if len(test_nodes) == 0 and len(low_degree_nodes) == 0:
        print(f"警告: test_nodes 和 low_degree_nodes 均为空，跳过结构扰动")
        return edge_index
    communities = data.y.to(device).cpu().numpy()
    while len(new_edges) < num_perturbations:
        u = np.random.choice(test_nodes if len(test_nodes) > 0 else low_degree_nodes)
        v = np.random.choice(low_degree_nodes)
        if u != v and communities[u] != communities[v]:
            new_edges.add((min(u, v), max(u, v)))
    new_edges = np.array(list(new_edges)).T
    perturbed_edge_index = tc.cat([remaining_edges, tc.tensor(new_edges, dtype=tc.long).to(device)], dim=1)
    return perturbed_edge_index


def enhanced_perturb_node_features(x, model, perturbation_ratio=0.3, epsilon=0.2, task_type='node_classification',
                                   mask=None):
    x = x.clone().detach().requires_grad_(True).to(device)
    output = model(x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask.to(device) & (
        mask if mask is not None else tc.ones_like(data.train_mask.to(device), dtype=bool))], data.y.to(device)[
                               data.train_mask.to(device) & (
                                   mask if mask is not None else tc.ones_like(data.train_mask.to(device), dtype=bool))])
    model.zero_grad()
    loss.backward()
    grad = x.grad
    feature_importance = tc.abs(grad).mean(dim=0)
    top_k_features = int(x.size(1) * perturbation_ratio * 2)
    important_features = tc.argsort(feature_importance, descending=True)[:top_k_features]

    feature_variance = tc.var(x, dim=0)
    adaptive_epsilon = epsilon * feature_variance / (feature_variance.mean() + 1e-8)
    if task_type == 'link_prediction':
        adaptive_epsilon *= 0.6

    perturbed_x = x.clone()
    for idx in important_features:
        noise = adaptive_epsilon[idx] * grad[:, idx].sign() + tc.randn_like(x[:, idx]) * 0.5
        perturbed_x[:, idx] += noise
    perturbed_x = tc.clamp(perturbed_x, min=x.min(), max=x.max())

    mean_diff = (x.mean() - perturbed_x.mean()).abs().item()
    var_diff = (x.var() - perturbed_x.var()).abs().item()
    if mean_diff > 0.1 or var_diff > 0.1:
        print(f"警告: 特征分布变化较大 (Mean Diff: {mean_diff:.4f}, Var Diff: {var_diff:.4f})")
    return perturbed_x


# 测试模型
def test(mask=None):
    model.eval()
    output = model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[(mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)] ==
               data.y.to(device)[
                   (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)]).sum()
    total = ((mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)).sum()
    acc = int(correct) / int(total)
    return acc


# 执行攻击并微调
def perform_attack_and_finetune(mask=None):
    perturbed_x, perturbed_edge_index = enhanced_stealthy_adversarial_attack(data, model, mask=mask)
    for epoch in range(100):
        loss = train(mask)
        if epoch % 10 == 0:
            acc = test(mask)
            print(f'微调轮次: {epoch:03d}, 损失: {loss:.4f}, 准确率: {acc:.4f}')
    return perturbed_x, perturbed_edge_index


# 计算置信度
def calculate_confidence(output, mask):
    prob = F.softmax(output[mask], dim=1)
    max_probs = prob.max(dim=1)[0]
    confidence = max_probs.mean().item()
    return confidence


# 评估指标
def evaluate_with_metrics(perturbed_x, perturbed_edge_index, mask=None):
    def evaluate_attack_metrics(perturbed_x, perturbed_edge_index, mask=None, results=None):
        original_pred = model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0),
                              data.edge_index).argmax(dim=1)
        perturbed_pred = model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0),
                               perturbed_edge_index).argmax(dim=1)

        asr = calculate_asr(original_pred, perturbed_pred, data.test_mask.to(device), data.y.to(device), mask)
        print(f'Attack Success Rate (ASR): {asr:.4f}')

        feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
        print(f'Perturbation Ratio: {feature_perturbation_ratio:.4f}')

        original_confidence = calculate_confidence(
            model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index),
            (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device))
        perturbed_confidence = calculate_confidence(
            model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index),
            (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device))
        confidence_drop = original_confidence - perturbed_confidence
        print(f'Original Confidence: {original_confidence:.4f}')
        print(f'Perturbed Confidence: {perturbed_confidence:.4f}')
        print(f'Confidence Drop: {confidence_drop:.4f}')

        y_true = data.y.to(device)[
            (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)].cpu().numpy()
        y_pred = perturbed_pred[
            (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)].cpu().numpy()
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        cm = confusion_matrix(y_true, y_pred)
        class_names = [str(i) for i in range(dataset.num_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='cividis_r', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Cora - Confusion Matrix (Sub-distribution: {mask is not None and "Masked" or "Full"})')
        plt.show()

        y_pred_probs = F.softmax(
            model(perturbed_x, perturbed_edge_index)[
                (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)],
            dim=1).detach().cpu().numpy()
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

        # 将 ASR 存储到 results
        if results is not None:
            results[sub_dist_name]['asr'] = asr


# 攻击后测试
def test_after_attack(perturbed_x, perturbed_edge_index, mask=None):
    model.eval()
    output = model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index)
    pred = output.argmax(dim=1)
    correct = (pred[(mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)] ==
               data.y.to(device)[
                   (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)]).sum()
    total = ((mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)).sum()
    acc = int(correct) / int(total)
    return acc


# 计算攻击成功率
def calculate_asr(original_pred, perturbed_pred, test_mask, y_true, mask=None):
    # 修复ASR计算：比较原始预测和攻击后预测的差异
    effective_mask = (mask & test_mask) if mask is not None else test_mask
    successful_attack = (perturbed_pred[effective_mask] != original_pred[effective_mask]).sum().item()
    total_attack = effective_mask.sum().item()
    asr = successful_attack / total_attack if total_attack > 0 else 0
    return asr


# 计算特征扰动比例
def calculate_feature_perturbation_ratio(original_x, perturbed_x):
    perturbation = (original_x - perturbed_x).abs()
    feature_perturbation_ratio = perturbation.mean().item()
    return feature_perturbation_ratio


# 攻击指标评估
def evaluate_attack_metrics(perturbed_x, perturbed_edge_index, mask=None, results=None):
    original_pred = model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0),
                          data.edge_index).argmax(dim=1)
    perturbed_pred = model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0),
                           perturbed_edge_index).argmax(dim=1)

    asr = calculate_asr(original_pred, perturbed_pred, data.test_mask.to(device), data.y.to(device), mask)
    print(f'Attack Success Rate (ASR): {asr:.4f}')

    feature_perturbation_ratio = calculate_feature_perturbation_ratio(data.x, perturbed_x)
    print(f'Perturbation Ratio: {feature_perturbation_ratio:.4f}')

    original_confidence = calculate_confidence(
        model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index),
        (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device))
    perturbed_confidence = calculate_confidence(
        model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index),
        (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device))
    confidence_drop = original_confidence - perturbed_confidence
    print(f'Original Confidence: {original_confidence:.4f}')
    print(f'Perturbed Confidence: {perturbed_confidence:.4f}')
    print(f'Confidence Drop: {confidence_drop:.4f}')

    y_true = data.y.to(device)[
        (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)].cpu().numpy()
    y_pred = perturbed_pred[
        (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)].cpu().numpy()
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
    plt.title(f'Cora - Confusion Matrix (Sub-distribution: {mask is not None and "Masked" or "Full"})')
    plt.show()

    y_pred_probs = F.softmax(model(perturbed_x, perturbed_edge_index)[
                                 (mask & data.test_mask.to(device)) if mask is not None else data.test_mask.to(device)],
                             dim=1).detach().cpu().numpy()
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

    # 将 ASR 存储到 results
    if results is not None:
        results[sub_dist_name]['asr'] = asr


# 资源使用评估
def evaluate_resource_usage(perturbed_x, perturbed_edge_index, mask=None):
    gpu_mem_usage = []
    cpu_mem_usage = []
    time_delays = []
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
    print("\n评估原始模型资源使用...")
    start_time = time.time()
    with tc.no_grad():
        model.eval()
        original_output = model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index)
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
        perturbed_output = model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0),
                                 perturbed_edge_index)
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
        train_loss = train(mask)
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
        plt.bar(['原始', '攻击后'], gpu_mem_usage)
        plt.title('GPU内存使用 (MB)')
        plt.ylabel('内存 (MB)')
    plt.subplot(1, 3, 2)
    plt.bar(['原始', '攻击后'], cpu_mem_usage)
    plt.title('CPU内存使用 (MB)')
    plt.ylabel('内存 (MB)')
    plt.subplot(1, 3, 3)
    plt.bar(['原始', '攻击后'], time_delays)
    plt.title('推理时间 (秒)')
    plt.ylabel('时间 (秒)')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 4))
    if tc.cuda.is_available():
        plt.subplot(1, 2, 1)
        plt.plot(train_mem_usage)
        plt.title('训练GPU内存使用')
        plt.xlabel('轮次')
        plt.ylabel('内存 (MB)')
    plt.subplot(1, 2, 2)
    plt.plot(train_time_usage)
    plt.title('每轮训练时间')
    plt.xlabel('轮次')
    plt.ylabel('时间 (秒)')
    plt.tight_layout()
    plt.show()


# 计算性能分析
def profile_computation(perturbed_x, perturbed_edge_index, mask=None):
    print("\n运行计算性能分析...")
    with tc.profiler.profile(
            activities=[tc.profiler.ProfilerActivity.CPU, tc.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_flops=True
    ) as prof:
        with tc.no_grad():
            model(data.x + (model.prompt_graph if model.prompt_graph is not None else 0), data.edge_index)
            model(perturbed_x + (model.prompt_graph if model.prompt_graph is not None else 0), perturbed_edge_index)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("gcn_performance_trace.json")
    print("性能分析跟踪已保存到 gcn_performance_trace.json")


# 跨分布测试主程序
if __name__ == "__main__":
    model.set_prompt_graph()
    for sub_dist_name, mask in sub_distributions.items():
        print(f"\n=== 开始测试子分布: {sub_dist_name} ===")
        mask = mask.bool().to(device)
        if mask.sum().item() == 0:
            print(f"错误: 子分布 {sub_dist_name} 的掩码为空，跳过！")
            continue

        try:
            # 预训练
            for epoch in range(100):
                loss = train_with_contrastive_loss(mask)
                if epoch % 10 == 0:
                    print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

            # 执行攻击并微调
            perturbed_x, perturbed_edge_index = perform_attack_and_finetune(mask)
            original_acc = test(mask)
            print(f'Original Accuracy (Before Attack): {original_acc:.4f}')
            perturbed_acc = test_after_attack(perturbed_x, perturbed_edge_index, mask)
            print(f'Perturbed Accuracy (After Attack): {perturbed_acc:.4f}')

            # 评估攻击指标，传递 results
            evaluate_attack_metrics(perturbed_x, perturbed_edge_index, mask, results)

            # 评估其他指标
            hits_at_k, mrr_score = evaluate_with_metrics(perturbed_x, perturbed_edge_index, mask)

            # 资源使用评估
            evaluate_resource_usage(perturbed_x, perturbed_edge_index, mask)

            # 计算性能分析（仅在 GPU 可用时执行）
            if tc.cuda.is_available():
                profile_computation(perturbed_x, perturbed_edge_index, mask)

            # 存储结果
            results[sub_dist_name] = {
                'original_acc': original_acc,
                'perturbed_acc': perturbed_acc,
                'asr': asr,
                'hits_at_k': hits_at_k['hits@50'],
                'mrr': mrr_score['mrr']
            }

        except Exception as e:
            print(f"子分布 {sub_dist_name} 处理时发生错误: {str(e)}")
            continue

# 生成三个子分布的ASR对比图
if results:
    print("\n=== 生成ASR对比图 ===")
    sub_dist_names = list(results.keys())
    asr_values = [results[name]['asr'] for name in sub_dist_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sub_dist_names, asr_values, color=['#f7a24f', '#ff184c', '#470b16'])
    plt.title('GCN+GraphPrompt 跨分布攻击成功率对比', fontsize=16, fontweight='bold')
    plt.xlabel('子分布类型', fontsize=12)
    plt.ylabel('攻击成功率 (ASR)', fontsize=12)
    plt.ylim(0, 1.0)

    # 在柱状图上添加数值标签
    for bar, value in zip(bars, asr_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印详细结果
    print("\n=== 详细结果汇总 ===")
    for name in sub_dist_names:
        print(f"{name}: ASR = {results[name]['asr']:.4f}, "
              f"原始准确率 = {results[name]['original_acc']:.4f}, "
              f"攻击后准确率 = {results[name]['perturbed_acc']:.4f}")
else:
    print("没有可用的结果数据生成对比图")
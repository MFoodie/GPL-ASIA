import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops, degree
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

# 定义子分布划分标准（基于节点度）
degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
degree_thresholds = [np.percentile(degrees, 33), np.percentile(degrees, 66)]  # 三分位数划分
sub_distributions = {
    'low_degree': tc.tensor(degrees <= degree_thresholds[0], dtype=bool).to(device),
    'mid_degree': tc.tensor((degrees > degree_thresholds[0]) & (degrees <= degree_thresholds[1]), dtype=bool).to(device),
    'high_degree': tc.tensor(degrees > degree_thresholds[1], dtype=bool).to(device)
}
results = {}

# 定义 GAE 编码器
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 定义完整的 GAE 模型
class GAE_Model(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GAE_Model, self).__init__()
        self.gae = GAE(GCNEncoder(nfeat, nhid))
        self.classifier = nn.Linear(nhid, dataset.num_classes)
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
        print(f'Prompt graph set with shape: {prompt_graph.shape}')

# 模型参数
nfeat = dataset.num_features
nhid = 16
dropout = 0.5
model = GAE_Model(nfeat, nhid, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 对比损失训练
def train_with_contrastive_loss(model, data, mask):
    model.train()
    optimizer.zero_grad()

    # 简化对比学习：只使用原始数据和轻微扰动
    z1 = model.gae.encode(data.x, data.edge_index)
    
    # 简单的数据增强而不是完整攻击
    noise = tc.randn_like(data.x) * 0.01
    x_aug = data.x + noise
    z2 = model.gae.encode(x_aug, data.edge_index)

    loss = contrastive_loss(z1[mask], z2[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 简化的对比损失计算
def contrastive_loss(z1, z2, temperature=0.5):
    # 简化计算，减少矩阵运算
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    # 只计算正样本对的相似度，避免全矩阵计算
    pos_sim = tc.sum(z1 * z2, dim=1) / temperature
    loss = -tc.log(tc.sigmoid(pos_sim)).mean()
    return loss

# 标准训练（微调）
def train(model, data, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y.to(device)[mask])
    loss.backward()
    optimizer.step()
    # 清理GPU内存
    if tc.cuda.is_available():
        tc.cuda.empty_cache()

    return loss.item()

# 测试函数
def test(model, data, mask=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    if mask is not None:
        correct = (pred[mask & data.test_mask.to(device)] == data.y.to(device)[mask & data.test_mask.to(device)]).sum()
        total = (mask & data.test_mask.to(device)).sum()
    else:
        correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
        total = data.test_mask.to(device).sum()
    acc = int(correct) / int(total)
    return acc

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
def crossba_perturb_node_features(x, model, edge_index, perturbation_ratio, epsilon, target_embedding, reg_param_1, reg_param_2, data):
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

def stealthy_adversarial_attack(data, model, epsilon=0.1, perturbation_ratio=0.3, num_trigger_nodes=5, task_weight=0.5, mask=None):
    model.eval()
    num_nodes = data.x.size(0) if mask is None else mask.sum().item()

    with tc.no_grad():
        prompt_embedding = model.prompt_graph if model.prompt_graph is not None else tc.zeros_like(data.x)

    prompt_norms = tc.norm(prompt_embedding, dim=1)
    node_scores = prompt_norms * tc.norm(data.x, dim=1)
    if mask is not None:
        node_scores = node_scores[mask]
        trigger_nodes = tc.topk(node_scores, min(num_trigger_nodes, mask.sum().item()))[1]
        trigger_nodes = tc.where(mask)[0][trigger_nodes]
    else:
        _, trigger_nodes = tc.topk(node_scores, num_trigger_nodes)

    edge_index = data.edge_index.clone()
    target_node = tc.randint(0, num_nodes, (1,)).item() if mask is None else tc.where(mask)[0][tc.randint(0, mask.sum().item(), (1,))].item()
    for node in trigger_nodes:
        new_edges = tc.tensor([[target_node, node.item()], [node.item(), target_node]], dtype=tc.long).to(device)
        edge_index = tc.cat([edge_index, new_edges], dim=1)

    subgraph_size = int(num_nodes * perturbation_ratio)
    if mask is None:
        _, subgraph_nodes = tc.topk(node_scores, subgraph_size)
    else:
        _, subgraph_nodes = tc.topk(node_scores, min(subgraph_size, mask.sum().item()))
        subgraph_nodes = tc.where(mask)[0][subgraph_nodes]
    for node in subgraph_nodes:
        for _ in range(3):
            target = np.random.choice(num_nodes) if mask is None else np.random.choice(tc.where(mask)[0].cpu().numpy())
            new_edges = tc.tensor([[node.item(), target], [target, node.item()]], dtype=tc.long).to(device)
            edge_index = tc.cat([edge_index, new_edges], dim=1)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    num_edges = edge_index.size(1)
    num_perturbations = int(num_edges * perturbation_ratio)

    edge_scores = tc.zeros(num_edges, device=edge_index.device).to(device)
    for i in range(num_edges):
        u, v = edge_index[:, i]
        edge_scores[i] = tc.dot(prompt_embedding[u], prompt_embedding[v])

    _, keep_indices = tc.topk(edge_scores, num_edges - num_perturbations)
    perturbed_edge_index = edge_index[:, keep_indices]

    new_edges = []
    for _ in range(num_perturbations):
        u, v = np.random.choice(num_nodes, 2, replace=False) if mask is None else np.random.choice(tc.where(mask)[0].cpu().numpy(), 2, replace=False)
        if u != v and tc.dot(prompt_embedding[u], prompt_embedding[v]) > 0:
            new_edges.append([u, v])
    if new_edges:
        new_edges = tc.tensor(new_edges, dtype=tc.long).to(device).t()
        perturbed_edge_index = tc.cat([perturbed_edge_index, new_edges], dim=1).to(device)

    # 生成目标嵌入用于CrossBA攻击
    target_embedding = generate_target_embedding(data, model, target_node)
    
    # 使用CrossBA多目标优化进行特征扰动
    perturbed_x = crossba_perturb_node_features(data.x, model, data.edge_index, 
                                               perturbation_ratio, epsilon, target_embedding, 
                                               reg_param_1, reg_param_2, data)
    
    if mask is not None:
        full_perturbed_x = data.x.clone()
        full_perturbed_x[mask] = perturbed_x[mask]
        perturbed_x = full_perturbed_x

    return perturbed_x.detach(), perturbed_edge_index

# 设置提示图
prompt_graph = tc.randn(data.num_nodes, nhid).to(device)
model.set_prompt_graph(prompt_graph)

# 攻击后测试
def test_after_attack(model, perturbed_x, perturbed_edge_index, data, mask=None):
    model.eval()
    out = model(perturbed_x, perturbed_edge_index)
    pred = out.argmax(dim=1)
    if mask is not None:
        correct = (pred[mask & data.test_mask.to(device)] == data.y.to(device)[mask & data.test_mask.to(device)]).sum()
        total = (mask & data.test_mask.to(device)).sum()
    else:
        correct = (pred[data.test_mask.to(device)] == data.y.to(device)[data.test_mask.to(device)]).sum()
        total = data.test_mask.to(device).sum()
    acc = int(correct) / int(total)
    return acc

# 链接预测评估
def evaluate_link_prediction(model, data, mask=None):
    model.gae.eval()
    z = model.gae.encode(data.x, data.edge_index)
    if mask is not None:
        z = z[mask]
        # 重新映射节点索引
        node_map = tc.where(mask)[0]
        # 过滤 pos_edge_index
        pos_edge_mask = tc.all(tc.isin(data.edge_index, node_map), dim=0)
        pos_edge_index = data.edge_index[:, pos_edge_mask]
        # 映射到子分布内的相对索引
        pos_edge_index = tc.stack([
            tc.searchsorted(node_map, pos_edge_index[0]),
            tc.searchsorted(node_map, pos_edge_index[1])
        ])
    else:
        pos_edge_index = data.edge_index

    neg_edge_index = tc.randint(0, z.size(0), (2, pos_edge_index.size(1)), device=device) if mask is not None else tc.randint(0, data.num_nodes, (2, pos_edge_index.size(1)), device=device)

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = tc.cat([pos_y, neg_y], dim=0).to(device)

    pos_pred = model.gae.decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = model.gae.decoder(z, neg_edge_index, sigmoid=True)
    pred = tc.cat([pos_pred, neg_pred], dim=0).to(device)

    auc = roc_auc_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy()) if len(y) > 0 and len(pred) > 0 else 0.0
    print(f'链接预测AUC: {auc:.4f}')
    return auc

def calculate_asr_and_specificity(original_pred, perturbed_pred, test_mask, target_class=None, mask=None):
    """
    计算攻击成功率 (ASR) 和特异度 (Specificity)
    """
    # 使用 test_mask & mask (如果 mask 存在) 确保一致性
    effective_mask = test_mask if mask is None else (test_mask & mask)
    original_labels = original_pred[effective_mask].cpu()
    perturbed_labels = perturbed_pred[effective_mask].cpu()
    true_labels = data.y.to(device)[effective_mask].cpu()

    if target_class is not None:
        success_mask = (original_labels != target_class) & (perturbed_labels == target_class)
    else:
        correct_original = (original_labels == true_labels)
        incorrect_perturbed = (perturbed_labels != true_labels)
        success_mask = correct_original & incorrect_perturbed

    asr = success_mask.sum().item() / max(1, correct_original.sum().item())

    if target_class is not None:
        tn_mask = (original_labels != target_class) & (perturbed_labels != target_class)
        fp_mask = (original_labels != target_class) & (perturbed_labels == target_class)
    else:
        tn_mask = (original_labels == true_labels) & (perturbed_labels == true_labels)
        fp_mask = (original_labels == true_labels) & (perturbed_labels != true_labels)

    specificity = tn_mask.sum().item() / max(1, (tn_mask.sum().item() + fp_mask.sum().item()))

    return asr, specificity

# 评估指标
def evaluate_attack_metrics(model, perturbed_x, perturbed_edge_index, data, mask=None):
    model.eval()
    original_out = model(data.x, data.edge_index)
    perturbed_out = model(perturbed_x, perturbed_edge_index)
    original_pred = original_out.argmax(dim=1)
    perturbed_pred = perturbed_out.argmax(dim=1)

    asr, specificity = calculate_asr_and_specificity(original_pred, perturbed_pred, data.test_mask.to(device), mask=mask)
    print(f'攻击成功率 (ASR): {asr:.4f}')
    print(f'特异度 (Specificity): {specificity:.4f}')

    if mask is not None:
        y_true = data.y.to(device)[data.test_mask.to(device) & mask].cpu().numpy()
        y_pred = perturbed_pred[data.test_mask.to(device) & mask].cpu().numpy()
    else:
        y_true = data.y.to(device)[data.test_mask.to(device)].cpu().numpy()
        y_pred = perturbed_pred[data.test_mask.to(device)].cpu().numpy()

    if len(y_true) > 0 and len(y_pred) > 0:
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'精确度: {precision:.4f}')
        print(f'召回率: {recall:.4f}')
        print(f'F1分数: {f1:.4f}')
    else:
        print("警告: 没有足够的样本进行评估指标计算.")
        precision, recall, f1 = 0.0, 0.0, 0.0

    # 不再绘制混淆矩阵
    return asr, specificity  # 添加返回值

# 资源使用评估
def evaluate_resource_usage(perturbed_x, perturbed_edge_index, data, mask=None):
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
        train_loss = train(model, data, mask)
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
def profile_computation(perturbed_x, perturbed_edge_index, data, mask=None):
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

# 跨分布测试主程序
for sub_dist_name, mask in sub_distributions.items():
    print(f"\n=== 测试子分布: {sub_dist_name} ===")
    mask = mask.bool()
    print(f"子分布节点数: {mask.sum().item()}")

    try:
        # 设置提示图
        prompt_graph = tc.randn(data.num_nodes, nhid).to(device)
        model.set_prompt_graph(prompt_graph)

        # 预训练（减少对比学习频率）
        for epoch in range(100):
            if epoch % 5 == 0:  # 只在每5轮使用对比学习
                loss = train_with_contrastive_loss(model, data, mask)
            else:  # 其他轮使用普通训练
                loss = train(model, data, mask)
            if epoch % 10 == 0:
                if tc.cuda.is_available():
                    gpu_mem = tc.cuda.memory_allocated() / 1024**2
                    print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}, GPU内存: {gpu_mem:.1f}MB')
                else:
                    print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # 微调
        for epoch in range(100):
            loss = train(model, data, mask)
            if epoch % 10 == 0:
                acc = test(model, data, mask)
                if tc.cuda.is_available():
                    gpu_mem = tc.cuda.memory_allocated() / 1024**2
                    print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, GPU内存: {gpu_mem:.1f}MB')
                else:
                    print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # 执行攻击并评估
        perturbed_x, perturbed_edge_index = stealthy_adversarial_attack(data, model, mask=mask)
        original_acc = test(model, data, mask)
        print(f'攻击前准确率: {original_acc:.4f}')
        perturbed_acc = test_after_attack(model, perturbed_x, perturbed_edge_index, data, mask)
        print(f'攻击后准确率: {perturbed_acc:.4f}')

        # 评估攻击指标
        asr, specificity = evaluate_attack_metrics(model, perturbed_x, perturbed_edge_index, data, mask)
        # 链接预测评估
        link_auc = evaluate_link_prediction(model, data, mask)
        # 资源使用评估
        evaluate_resource_usage(perturbed_x, perturbed_edge_index, data, mask)
        # 计算性能分析（仅在 GPU 可用时执行）
        if tc.cuda.is_available():
            profile_computation(perturbed_x, perturbed_edge_index, data, mask)

        # 存储结果
        results[sub_dist_name] = {
            'original_acc': original_acc,
            'perturbed_acc': perturbed_acc,
            'asr': asr,
            'link_auc': link_auc
        }
    except Exception as e:
        print(f"子分布 {sub_dist_name} 处理失败: {str(e)}")
        continue

# 汇总跨分布结果
print("\n=== 跨分布测试汇总 ===")
for sub_dist_name, result in results.items():
    print(f"子分布 {sub_dist_name}:")
    print(f"  攻击前准确率: {result['original_acc']:.4f}")
    print(f"  攻击后准确率: {result['perturbed_acc']:.4f}")
    print(f"  攻击成功率 (ASR): {result['asr']:.4f}")
    print(f"  链接预测AUC: {result['link_auc']:.4f}")

# 绘制三个子分布 ASR 对比图
if len(results) > 0:
    order = ['low_degree', 'mid_degree', 'high_degree']
    colors = ['#f7a24f', '#ff184c', '#470b16']
    labels, values = [], []
    for k in order:
        if k in results:
            labels.append(k)
            values.append(results[k]['asr'])
    if len(values) > 0:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=colors[:len(values)])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
        plt.title('Cora - 子分布 ASR 对比')
        plt.ylabel('ASR')
        plt.xlabel('子分布')
        plt.tight_layout()
        plt.show()

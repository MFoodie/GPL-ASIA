import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import random
import os
import time
import torch.cuda as cuda
import psutil
from torch_geometric.utils import degree

# 设置随机种子
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

# 加载数据集
dataset = TUDataset(root='D:/SRTP/SRTP/data', name='ENZYMES', force_reload=False)
dataset = dataset.shuffle()
train_loader = DataLoader(dataset[:480], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[480:], batch_size=32, shuffle=False)

# GCN模型定义
class GCN(nn.Module):  # 类名改为GCN
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()
        # 将GATConv改为GCNConv，注意GCN不需要多头注意力机制
        self.conv1 = GCNConv(nfeat, nhid)  # 移除了heads参数
        self.conv2 = GCNConv(nhid, nhid)   # 移除了heads参数
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))  # 保持相同的激活函数
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index, batch):
        x = self.encode(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 模型参数 - 针对GCN优化
nfeat = dataset.num_features
nhid = 32  # 增加隐藏层维度，提高模型容量
nclass = dataset.num_classes
dropout = 0.3  # 降低dropout，提高模型学习能力

model = GCN(nfeat, nclass, nhid, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # 降低学习率，增加权重衰减
criterion = nn.CrossEntropyLoss()

# 对比学习损失函数
def contrastive_loss(z1, z2, temperature=0.1):
    """
    对比学习损失函数
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = tc.matmul(z1, z2.t()) / temperature
    
    # 对角线元素为正样本对
    labels = tc.arange(z1.size(0)).to(z1.device)
    
    # 计算对比损失
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def train_with_contrastive_loss_simple(data_loader):
    """
    增强的对比学习预训练，使用多样化数据增强提高攻击成功率，保持纯黑盒方法
    """
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
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
        
        # 获取图表示
        z1 = global_mean_pool(model.encode(x1, e1), data.batch)
        z2 = global_mean_pool(model.encode(x2, e2), data.batch)
        
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def optimize_attack_independently(data_loader, epochs=15):
    """
    独立的攻击优化阶段
    """
    print("开始攻击优化...")
    best_attack_data = None
    best_asr = 0.0
    
    for epoch in range(epochs):
        model.eval()
        total_asr = 0.0
        attack_data_list = []
        
        for data in data_loader:
            data = data.to(device)
            
            # 执行攻击
            attacked_data = stealthy_adversarial_attack(data, model, attack_round=epoch)
            attack_data_list.append(attacked_data)
            
            # 计算攻击成功率
            with tc.no_grad():
                original_output = model(data.x, data.edge_index, data.batch)
                attacked_output = model(attacked_data.x, attacked_data.edge_index, attacked_data.batch)
                
                original_pred = original_output.argmax(dim=1)
                attacked_pred = attacked_output.argmax(dim=1)
                
                asr = (original_pred != attacked_pred).float().mean().item()
                total_asr += asr
        
        avg_asr = total_asr / len(data_loader)
        
        if avg_asr > best_asr:
            best_asr = avg_asr
            best_attack_data = attack_data_list
        
        if epoch % 5 == 0:
            print(f'Attack Optimization Epoch: {epoch:03d}, ASR: {avg_asr:.4f}')
    
    print(f'Best ASR: {best_asr:.4f}')
    return best_attack_data

def perform_attack_and_finetune(data_loader, attack_data_list, epochs=30):
    """
    执行攻击并进行微调
    """
    print("开始微调...")
    
    # 创建攻击数据加载器
    attack_loader = DataLoader(attack_data_list, batch_size=data_loader.batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in attack_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Finetune Epoch: {epoch:03d}, Loss: {total_loss/len(attack_loader):.4f}')
    
    return attack_data_list

# 预训练阶段（简化版本，用于对比学习）
def pretrain(data_loader, epochs=80):
    print("开始预训练...")
    train_losses = []
    
    for epoch in range(epochs):
        loss = train_with_contrastive_loss_simple(data_loader)
        train_losses.append(loss)
        
        if epoch % 10 == 0:
            if tc.cuda.is_available():
                gpu_mem = tc.cuda.memory_allocated() / 1024**2
                print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}, GPU内存: {gpu_mem:.1f}MB')
            else:
                print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.title('Contrastive Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

# CrossBA纯黑盒动态自适应攻击方法
def enhanced_perturb_node_features(x, attack_node_ids, attack_strategy='adaptive', noise_scale=0.1):
    """
    增强的节点特征扰动，针对GCN优化
    """
    perturbed_x = x.clone()
    
    if attack_strategy == 'adaptive':
        # 基于特征方差的重要性计算（黑盒方法）
        feature_variance = x.var(dim=0)
        feature_importance = feature_variance / (feature_variance.sum() + 1e-8)
        
        for node_id in attack_node_ids:
            # 针对GCN的卷积操作，使用适中的扰动
            importance_weights = feature_importance.unsqueeze(0)
            # 增加扰动强度，特别是对重要特征
            enhanced_noise_scale = noise_scale * 1.25  # 增加25%的扰动强度
            noise = tc.randn_like(x[node_id]) * enhanced_noise_scale * importance_weights
            
            # 添加特征缩放扰动，影响卷积学习
            scale_factor = 0.9 + tc.rand(1).item() * 0.2  # 0.9-1.1的随机缩放
            perturbed_x[node_id] = x[node_id] * scale_factor + noise
            
    elif attack_strategy == 'conv_focused':
        # 针对GCN卷积操作的专门扰动策略
        for node_id in attack_node_ids:
            # 对特征进行适度的扰动，影响卷积计算
            conv_noise = tc.randn_like(x[node_id]) * noise_scale * 1.6
            # 添加特征偏移扰动，影响邻居聚合
            offset = tc.randn_like(x[node_id]) * noise_scale * 0.5
            perturbed_x[node_id] = x[node_id] + conv_noise + offset
            
    elif attack_strategy == 'degree_based':
        # 基于度的扰动策略
        for node_id in attack_node_ids:
            noise = tc.randn_like(x[node_id]) * noise_scale * 1.1
            perturbed_x[node_id] = x[node_id] + noise
            
    else:  # random
        # 随机扰动，增加强度
        for node_id in attack_node_ids:
            noise = tc.randn_like(x[node_id]) * noise_scale * 1.15
            perturbed_x[node_id] = x[node_id] + noise
    
    return perturbed_x

def perturb_node_features(x, attack_node_ids, noise_scale=0.1):
    """
    节点特征扰动，纯黑盒方法
    """
    perturbed_x = x.clone()
    
    # 基于统计特征重要性的扰动
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0)
    
    for node_id in attack_node_ids:
        # 使用特征统计信息生成扰动
        importance = feature_std / (feature_std.sum() + 1e-8)
        noise = tc.randn_like(x[node_id]) * noise_scale * importance
        perturbed_x[node_id] = x[node_id] + noise
    
    return perturbed_x

def add_enhanced_trigger_nodes(x, edge_index, attack_node_ids, pattern='multi_nodes', num_triggers=2):
    """
    增强的触发节点添加，支持多种模式
    """
    new_nodes = []
    new_edges = []
    current_nodes = x.size(0)
    
    if pattern == 'multi_nodes':
        # 多节点触发模式
        for node_id in attack_node_ids:
            for _ in range(num_triggers):
                # 基于目标节点特征生成触发节点
                target_feat = x[node_id]
                trigger_feat = target_feat + tc.randn_like(target_feat) * 0.1
                new_nodes.append(trigger_feat)
                new_node_id = current_nodes + len(new_nodes) - 1
                new_edges.append([node_id, new_node_id])
                
    elif pattern == 'trigger_graph':
        # 触发子图模式
        for node_id in attack_node_ids:
            # 创建小型触发子图
            for i in range(num_triggers):
                trigger_feat = x[node_id] + tc.randn_like(x[node_id]) * 0.15
                new_nodes.append(trigger_feat)
                new_node_id = current_nodes + len(new_nodes) - 1
                new_edges.append([node_id, new_node_id])
                
                # 在触发节点之间添加连接
                if i > 0:
                    prev_trigger_id = current_nodes + len(new_nodes) - 2
                    new_edges.append([prev_trigger_id, new_node_id])
    
    if new_nodes:
        x = tc.cat([x, tc.stack(new_nodes).to(x.device)], dim=0)
        new_edges = tc.tensor(new_edges, dtype=tc.long, device=edge_index.device).t()
        edge_index = tc.cat([edge_index, new_edges], dim=1)
    
    return x, edge_index

def adaptive_attack_strategy(model, data, attack_round=0):
    """
    动态自适应攻击策略，针对GCN优化
    """
    model.eval()
    with tc.no_grad():
        output = model(data.x, data.edge_index, data.batch)
        confidence = F.softmax(output, dim=1).max(dim=1)[0].mean().item()
        
        # 计算不确定性（熵）
        entropy = -(F.softmax(output, dim=1) * F.log_softmax(output, dim=1)).sum(dim=1).mean().item()
        
        # 计算图结构复杂度
        degrees = degree(data.edge_index[0], num_nodes=data.x.size(0))
        degree_variance = degrees.float().var().item()
    
    # 针对GCN的动态调整攻击参数（保守但有效策略）
    if confidence > 0.8:
        noise_scale = 0.18 + attack_round * 0.02  # 适中的扰动强度
        attack_ratio = 0.32 + attack_round * 0.04  # 适中的攻击节点比例
    elif confidence > 0.6:
        noise_scale = 0.13 + attack_round * 0.015
        attack_ratio = 0.22 + attack_round * 0.03
    else:
        noise_scale = 0.08 + attack_round * 0.01
        attack_ratio = 0.12 + attack_round * 0.02
    
    # 根据不确定性调整（GCN对不确定性相对稳定）
    if entropy > 1.0:  # 高不确定性
        noise_scale *= 1.25  # 适度增加扰动强度
        attack_ratio *= 1.1
    elif entropy < 0.5:  # 低不确定性，需要适度攻击
        noise_scale *= 1.15
        attack_ratio *= 1.05
    
    # 根据图复杂度调整
    if degree_variance > 10.0:  # 复杂图结构
        noise_scale *= 0.9  # 适度降低
        attack_ratio *= 0.95
    else:  # 简单图结构，可以适度激进
        noise_scale *= 1.08
        attack_ratio *= 1.03
    
    # 针对GCN的额外调整
    noise_scale *= 1.1  # 整体增加10%的扰动强度
    attack_ratio *= 1.05  # 整体增加5%的攻击节点比例
    
    return min(noise_scale, 0.35), min(attack_ratio, 0.55)  # 适中的上限

def stealthy_adversarial_attack(data, model, attack_round=0):
    """
    隐蔽对抗攻击，基于CrossBA的纯黑盒方法
    """
    data = data.clone()
    
    # 动态选择攻击策略
    noise_scale, attack_ratio = adaptive_attack_strategy(model, data, attack_round)
    
    # 选择攻击节点
    num_nodes = data.x.size(0)
    num_attack_nodes = int(num_nodes * attack_ratio)
    
    # 基于度的节点选择（更自然）
    degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    sorted_indices = tc.argsort(degrees, descending=True)
    attack_node_ids = sorted_indices[:num_attack_nodes].tolist()
    
    # 特征扰动 - 针对GCN使用卷积聚焦策略
    attack_strategy = 'conv_focused' if attack_round % 5 == 0 else 'adaptive'
    perturbed_x = enhanced_perturb_node_features(
        data.x, attack_node_ids, 
        attack_strategy=attack_strategy, 
        noise_scale=noise_scale
    )
    
    # 结构扰动 - 添加触发节点
    pattern = 'multi_nodes' if attack_round % 2 == 0 else 'trigger_graph'
    perturbed_x, perturbed_edge_index = add_enhanced_trigger_nodes(
        perturbed_x, data.edge_index, attack_node_ids, 
        pattern=pattern, num_triggers=2
    )
    
    # 更新batch信息
    num_new = perturbed_x.size(0) - data.batch.size(0)
    if num_new > 0:
        new_batch = data.batch.new_full((num_new,), data.batch[0].item())
        data.batch = tc.cat([data.batch, new_batch], dim=0)
    
    data.x = perturbed_x
    data.edge_index = perturbed_edge_index
    
    return data

def create_perturbed_loader(data_loader, model, attack_round=0):
    """
    创建对抗样本数据加载器
    """
    perturbed_data = []
    for data in data_loader:
        perturbed_data.append(stealthy_adversarial_attack(data, model, attack_round))
    
    return DataLoader(perturbed_data, batch_size=data_loader.batch_size, shuffle=True)

# 评估函数
def evaluate(data_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for data in data_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)
        correct += (pred == data.y.to(device)).sum().item()
        total += data.y.to(device).size(0)
        y_true.extend(data.y.to(device).cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    acc = correct / total
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 确保混淆矩阵的标签范围与实际类别一致
    cm = confusion_matrix(y_true, y_pred, labels=range(nclass))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis_r', vmin=0, vmax=cm.max())  # vmin=0 确保最小值从0开始
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return acc, precision, recall, f1

def evaluate_attack(original_loader, perturbed_loader, device):
    original_pred, perturbed_pred, y_true = [], [], []

    model.eval()
    with tc.no_grad():
        for data in original_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            original_pred.append(output.argmax(dim=1))
            y_true.append(data.y.to(device))

        for data in perturbed_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            perturbed_pred.append(output.argmax(dim=1))

    original_pred = tc.cat(original_pred).to(device)
    perturbed_pred = tc.cat(perturbed_pred).to(device)
    y_true = tc.cat(y_true).to(device)

    # 修复ASR计算：比较原始预测和攻击后预测的差异
    asr = (original_pred != perturbed_pred).float().mean().item()
    acc = (perturbed_pred == y_true).float().mean().item()

    # 计算置信度
    def get_confidence(output):
        prob = F.softmax(output, dim=1)
        return prob.max(dim=1)[0].mean().item()

    sample_data = next(iter(original_loader)).to(device)
    original_conf = get_confidence(model(sample_data.x, sample_data.edge_index, sample_data.batch))

    sample_perturbed = next(iter(perturbed_loader)).to(device)
    perturbed_conf = get_confidence(model(sample_perturbed.x, sample_perturbed.edge_index, sample_perturbed.batch))

    print(f"\nAttack Evaluation:")
    print(f"Attack Success Rate: {asr:.4f}")
    print(f"Accuracy after attack: {acc:.4f}")
    print(f"Original Confidence: {original_conf:.4f}")
    print(f"Perturbed Confidence: {perturbed_conf:.4f}")
    print(f"Confidence Drop: {original_conf - perturbed_conf:.4f}")

    return asr, acc

# 资源评估
def evaluate_resources(model, loader, device, mode='inference'):
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
        tc.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    baseline_cpu = process.memory_info().rss / (1024 ** 2)

    start_time = time.time()

    if mode == 'inference':
        model.eval()
        with tc.no_grad():
            for data in loader:
                _ = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
    else:  # training
        model.train()
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y.to(device))
            loss.backward()
            optimizer.step()

    elapsed = time.time() - start_time
    current_cpu = process.memory_info().rss / (1024 ** 2) - baseline_cpu
    gpu_mem = cuda.max_memory_allocated(device) / (1024 ** 2) if tc.cuda.is_available() else 0

    return {
        'time': elapsed,
        'cpu_mem': current_cpu,
        'gpu_mem': gpu_mem
    }

def visualize_resources(orig_res, pert_res, train_res):
    labels = ['Original', 'Perturbed', 'Training']
    times = [orig_res['time'], pert_res['time'], train_res['time']]
    cpu_mems = [orig_res['cpu_mem'], pert_res['cpu_mem'], train_res['cpu_mem']]
    gpu_mems = [orig_res['gpu_mem'], pert_res['gpu_mem'], train_res['gpu_mem']] if tc.cuda.is_available() else []

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.bar(labels, times)
    plt.title('Time Consumption (s)')

    plt.subplot(1, 3, 2)
    plt.bar(labels, cpu_mems, color='orange')
    plt.title('CPU Memory (MB)')

    if gpu_mems:
        plt.subplot(1, 3, 3)
        plt.bar(labels, gpu_mems, color='green')
        plt.title('GPU Memory (MB)')

    plt.tight_layout()
    plt.show()

def test():
    """测试原始模型性能"""
    model.eval()
    correct = 0
    total = 0
    with tc.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            correct += (pred == data.y.to(device)).sum().item()
            total += data.y.to(device).size(0)
    return correct / total

def test_after_attack(attack_data_list):
    """测试攻击后的模型性能"""
    model.eval()
    correct = 0
    total = 0
    with tc.no_grad():
        for data in attack_data_list:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            correct += (pred == data.y.to(device)).sum().item()
            total += data.y.to(device).size(0)
    return correct / total

def evaluate_attack_metrics(attack_data_list):
    """评估攻击指标"""
    model.eval()
    original_preds = []
    attacked_preds = []
    y_true = []
    
    with tc.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            original_output = model(data.x, data.edge_index, data.batch)
            original_pred = original_output.argmax(dim=1)
            original_preds.extend(original_pred.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
            
            if i < len(attack_data_list):
                attacked_data = attack_data_list[i].to(device)
                attacked_output = model(attacked_data.x, attacked_data.edge_index, attacked_data.batch)
                attacked_pred = attacked_output.argmax(dim=1)
                attacked_preds.extend(attacked_pred.cpu().numpy())
    
    # 确保所有列表长度一致
    min_len = min(len(original_preds), len(attacked_preds), len(y_true))
    original_preds = original_preds[:min_len]
    attacked_preds = attacked_preds[:min_len]
    y_true = y_true[:min_len]
    
    # 计算攻击成功率
    asr = sum(1 for i in range(min_len) if original_preds[i] != attacked_preds[i]) / min_len
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    
    # 计算置信度下降
    original_conf = sum(1 for i in range(min_len) if original_preds[i] == y_true[i]) / min_len
    attacked_conf = sum(1 for i in range(min_len) if attacked_preds[i] == y_true[i]) / min_len
    conf_drop = original_conf - attacked_conf
    print(f"Confidence Drop: {conf_drop:.4f}")
    
    return asr, conf_drop

def evaluate_with_metrics(attack_data_list):
    """使用多种指标评估"""
    model.eval()
    y_true = []
    y_pred = []
    
    with tc.no_grad():
        for data in attack_data_list:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # 计算各种指标
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return precision, recall, f1, accuracy

def evaluate_resource_usage(attack_data_list):
    """评估资源使用情况"""
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
        tc.cuda.reset_peak_memory_stats(device)
    
    process = psutil.Process()
    baseline_cpu = process.memory_info().rss / (1024 ** 2)
    
    start_time = time.time()
    
    # 测试攻击数据推理
    model.eval()
    with tc.no_grad():
        for data in attack_data_list:
            data = data.to(device)
            _ = model(data.x, data.edge_index, data.batch)
    
    elapsed = time.time() - start_time
    current_cpu = process.memory_info().rss / (1024 ** 2) - baseline_cpu
    gpu_mem = cuda.max_memory_allocated(device) / (1024 ** 2) if tc.cuda.is_available() else 0
    
    print(f"Inference Time: {elapsed:.2f}s")
    print(f"CPU Memory: {current_cpu:.2f}MB")
    print(f"GPU Memory: {gpu_mem:.2f}MB" if tc.cuda.is_available() else "GPU Memory: N/A")

def profile_computation(attack_data_list):
    """计算性能分析"""
    if not tc.cuda.is_available():
        print("CUDA not available for profiling")
        return
    
    model.eval()
    with tc.no_grad():
        for data in attack_data_list[:5]:  # 只分析前5个样本
            data = data.to(device)
            with tc.cuda.profiler.profile() as prof:
                _ = model(data.x, data.edge_index, data.batch)
    
    print("Computation profiling completed")

def main():
    print("开始CrossBA纯黑盒动态自适应攻击...")
    
    # 第一阶段：纯对比学习预训练（不执行攻击，参考CrossBA的400 epochs）
    pretrain(train_loader, epochs=80)
    
    # 第二阶段：独立的攻击优化（参考CrossBA的10个epoch）
    attack_data_list = optimize_attack_independently(train_loader, epochs=15)
    
    # 第三阶段：微调（参考CrossBA的20-30个epoch）
    attack_data_list = perform_attack_and_finetune(train_loader, attack_data_list, epochs=30)
    
    # 评估攻击效果
    original_acc = test()
    print(f'Original Accuracy (Before Attack): {original_acc:.4f}')
    
    attacked_acc = test_after_attack(attack_data_list)
    print(f'Attacked Accuracy (After Attack): {attacked_acc:.4f}')
    
    # 评估攻击指标
    evaluate_attack_metrics(attack_data_list)
    evaluate_with_metrics(attack_data_list)
    evaluate_resource_usage(attack_data_list)
    
    if tc.cuda.is_available():
        profile_computation(attack_data_list)

if __name__ == '__main__':
    main()

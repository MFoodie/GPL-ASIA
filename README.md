# 面向图提示学习的隐蔽式对抗注入攻击

---

## 1. 项目概述

本项目展示了 **面向图提示学习的隐蔽式对抗注入攻击**（Graph Prompt Learning - Adversarial Stealthy Injection Attack, **GPL-ASIA**） 的多种算法及其实验结果，涵盖了经典图神经网络模型（GAT、GCN、GAE）在图提示学习下的对抗攻击与防御。创新点包括：
- 以运行时间和内存占用衡量攻击效果
- 攻击方法扩展至加权图
- 动态调整的自适应机制
- 三种跨上下文场景（跨类别、跨分布、跨数据集）下的测试
- GAT模型的防御实验

---

## 2. 目录结构与功能说明

| 文件夹/文件名           | 说明 |
|--------------------|------|
| graph_classify     | 图分类任务相关代码（如ENZYMES数据集） |
| node_classify      | 节点分类任务相关代码（如Cora、CiteSeer等） |
| innovative_point   | 项目创新点相关代码（如加权图、动态自适应机制） |
| cross_category     | 跨类别场景下的测试代码 |
| cross_distribution | 跨分布场景下的测试代码 |
| cross_database     | 跨数据集场景下的测试代码 |
| data/              | 存放各类原始数据集（Cora、Citeseer、PubMed、ENZYMES等） |
| outputs/           | 运行结果输出（如图片、日志等） |
| tools/             | 辅助工具脚本（如打包、可视化等） |
| gui_launcher.py    | 图形界面启动脚本 |
| requirements.txt   | Python依赖包列表 |
| build_icon.ico/png/svg | 项目图标文件（用于可执行文件打包） |
| dist/GPL-ASIA | 可执行`GPL-ASIA.exe`文件存放位置 |

---

## 3. 主要代码文件说明

### graph_classify（图分类任务）
- `GAE_graph_GraphPrompt_attack.py`  GAE+GraphPrompt方法攻击
- `GAE_graph_ProG_attack.py`         GAE+ProG方法攻击
- `GAT_graph_GraphPrompt_attack.py`  GAT+GraphPrompt方法攻击
- `GAT_graph_ProG_attack.py`         GAT+ProG方法攻击
- `GCN_graph_GraphPrompt_attack.py`  GCN+GraphPrompt方法攻击
- `GCN_graph_ProG_attack.py`         GCN+ProG方法攻击

### node_classify（节点分类任务）
- `evaluator.py`                     评估指标计算（MRR、Hits@K）
- `GAE_node_GraphPrompt_dynamic.py`  GAE+GraphPrompt动态自适应攻击
- `GAE_node_ProG_dynamic.py`         GAE+ProG动态自适应攻击
- `GAT_node_GraphPrompt_dynamic.py`  GAT+GraphPrompt动态自适应攻击
- `GAT_node_ProG_dynamic.py`         GAT+ProG动态自适应攻击
- `GCN_node_GraphPrompt_dynamic.py`  GCN+GraphPrompt动态自适应攻击
- `GCN_node_ProG_dynamic.py`         GCN+ProG动态自适应攻击
- `weighted_GAE.py`                  GAE加权图攻击
- `weighted_GAT.py`                  GAT加权图攻击
- `weighted_GCN.py`                  GCN加权图攻击

### cross_category（跨类别测试任务）
- `evaluator.py`                     评估指标计算
- `test1_GAT_GraphPrompt.py`         GAT+GraphPrompt跨场景测试
- `test2_GAT_ProG.py`                GAT+ProG跨场景测试
- `test3_GCN_GraphPrompt.py`         GCN+GraphPrompt跨场景测试
- `test4_GCN_ProG.py`                GCN+ProG跨场景测试
- `test5_GAE_GraphPrompt.py`         GAE+GraphPrompt跨场景测试
- `test6_GAE_ProG.py`                GAE+ProG跨场景测试

### cross_database（跨数据集测试任务）
- `evaluator.py`                     评估指标计算
- `test1_GAT_GraphPrompt.py`         GAT+GraphPrompt跨场景测试
- `test2_GAT_ProG.py`                GAT+ProG跨场景测试
- `test3_GCN_GraphPrompt.py`         GCN+GraphPrompt跨场景测试
- `test4_GCN_ProG.py`                GCN+ProG跨场景测试
- `test5_GAE_GraphPrompt.py`         GAE+GraphPrompt跨场景测试
- `test6_GAE_ProG.py`                GAE+ProG跨场景测试

### cross_distribution（跨分布测试任务）
- `evaluator.py`                     评估指标计算
- `test1_GAT_GraphPrompt.py`         GAT+GraphPrompt跨场景测试
- `test2_GCN_GraphPrompt.py`         GCN+GraphPrompt跨场景测试
- `test3_GAE_GraphPrompt.py`         GAE+GraphPrompt跨场景测试
- `test4_GAT_ProG.py`                GAT+ProG跨场景测试
- `test5_GCN_ProG.py`                GCN+ProG跨场景测试
- `test6_GAE_ProG.py`                GAE+ProG跨场景测试

### tools
- `matplotlib_hook.py`               matplotlib相关hook脚本
- 其它辅助脚本

### 其它
- `gui_launcher.py`                  图形界面启动脚本
- `requirements.txt`                 依赖包列表
- `build_icon.ico/png/svg`           项目图标

---

## 4. 环境配置与运行方法

1. **环境准备**
   - 推荐使用Python 3.8+，建议使用虚拟环境（如venv）
   - 安装依赖：
     ```bash
     pip install -r requirements.txt
     ```
   - 需提前安装PyTorch（版本建议见requirements.txt或官网）

2. **数据集准备**
   - 节点分类任务：Cora、CiteSeer、PubMed等，下载后放入`data/`目录下对应子文件夹
   - 图分类任务：ENZYMES等
   - 若需手动下载Cora/CiteSeer/PubMed数据集，可参考：https://github.com/kimiyoung/planetoid/tree/master/data

3. **运行示例**
   - 命令行运行：
     ```bash
     python graph_classify/GAE_graph_GraphPrompt_attack.py
     ```
   - 启动GUI：
     ```bash
     python gui_launcher.py
     ```

4. **可执行文件打包（可选）**
   - 使用PyInstaller等工具，推荐在虚拟环境下打包，需指定icon文件：
     ```bash
     pyinstaller --icon build_icon.ico gui_launcher.py
     ```

---

## 5. 其他说明
- 运行前请根据实际路径修改代码中的数据集路径参数
- outputs/目录下为运行结果图片、日志等
- 若遇依赖或环境问题，请优先检查requirements.txt和PyTorch安装
- 数据集默认下载在你的`D:/SRTP/SRTP/data`文件夹路径下，如需修改路径，可修改对应代码文件的此行代码（数据集名称可能为 Cora、CiteSeer、PubMed或ENZYMES）的数据集路径 `dataset = Planetoid(root='D:/SRTP/SRTP/data', name='Cora', force_reload=False)`，改为你的本地指定路径。
---

## 6. 参考
- [Planetoid数据集下载](https://github.com/kimiyoung/planetoid/tree/master/data)
- 相关论文与文档

---

## 7. 需要的主要依赖库

本项目主要依赖以下Python库（详见 requirements.txt）：

- torch（PyTorch）
- torch geometric
- numpy
- pandas
- matplotlib
- scikit-learn
- networkx
- tqdm
- cairosvg 或 pillow（部分可视化/图标功能）
- pyinstaller（如需打包为exe）

请使用如下命令一键安装：
```bash
pip install -r requirements.txt
```
如遇依赖问题，请根据报错信息手动安装缺失的库。

---

如有问题请联系作者（邮箱1436301457@qq.coms）。
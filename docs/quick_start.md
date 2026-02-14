# 快速开始

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

## 数据准备

1. **准备Criteo数据集**
   ```bash
   # 将Criteo数据放在 data/raw/criteo/train.txt
   ```

2. **数据分割**
   ```bash
   # 三段式数据分割：train/val/eval (90%/1%/9%)，包含shuffle
   bash scripts/prepare_criteo.sh data/raw/criteo/train.txt data/raw/criteo
   ```

## 训练流程

### 1. 配置模型参数

编辑 `configs/dnn_criteo/dnn.yaml`:

```yaml
# Embedding MLP Baseline Configuration
model:
  name: "embmlp"                          # 模型类型标识
  embedding_dim: 16                       # 类别特征嵌入维度
  hidden_dims: [256, 128, 64, 32]         # MLP隐藏层维度列表
  dropout: 0.2                            # Dropout正则化概率
  hash_size: 1000000                      # 类别特征哈希空间大小

# Dataset Configuration
data:
  dataset: "criteo"                                    # 数据集类型
  train_path: "data/raw/criteo/train_split.txt"  # 训练数据路径
  val_path: "data/raw/criteo/val_split.txt"      # 验证数据路径
  test_paths:                                          # 测试数据路径列表
    - "data/raw/criteo/eval_split.txt"           # 评估集路径
  batch_size: 2048                                    # 训练批处理大小
  eval_batch_size: 4096                              # 评估批处理大小
  use_sharding: true                                  # 启用多进程分片数据加载
  num_workers: 8                                      # 数据加载进程数

# Training Configuration
training:
  epochs: 1                   # 训练轮数
  learning_rate: 0.001        # 学习率
  optimizer: "adam"           # 优化器类型
  train_metric_interval: 100  # 训练过程中评估间隔步数

# Output Configuration
output:
  model_dir: "outputs_criteo/checkpoints/dnn"  # 模型保存目录
  log_dir: "outputs_criteo/logs/dnn"           # 日志保存目录
  save_last: true                              # 是否保存最新模型

# Evaluation Configuration
evaluation:
  metrics: ["auc", "logloss"]  # 评估指标列表
```

### 2. 开始训练

```bash
# 完整训练（使用完整criteo数据集）
python scripts/train.py --config configs/dnn_criteo/dnn.yaml --device auto
```

### 3. 模型评估

```bash
python scripts/evaluate.py \
  --config configs/dnn_criteo/dnn.yaml \
  --checkpoint outputs_criteo/checkpoints/dnn/best_checkpoint.pth \
  --device auto
```

### 4. 模型分析

训练过程自动集成了全面的模型分析，包括：
- **参数统计**: 区分embedding参数和其他参数
- **内存占用**: 运行时内存和磁盘存储分析
- **计算复杂度**: FLOPs计算和推理速度估算
- **文件大小**: checkpoint文件大小分析

模型分析在训练前后自动显示，也可单独使用：
```python
from utils.model_analysis import print_model_analysis
print_model_analysis(model, checkpoint_path="path/to/checkpoint.pth")
```

### 5. 输出结果

训练完成后，模型和日志保存在：
- **模型文件**: `outputs_criteo/checkpoints/dnn/`
  - `best_checkpoint.pth` - 最佳模型
  - `last_checkpoint.pth` - 最新模型
- **训练日志**: `outputs_criteo/logs/dnn/`
  - TensorBoard事件文件

## 性能优化建议

### 数据加载优化
- 启用多进程分片: `use_sharding: true`
- 调整worker数量: `num_workers: 8`
- 优化批处理大小: `batch_size: 2048`

### GPU利用率
- 使用自动设备选择: `--device auto`
- 监控GPU内存使用情况
- 根据GPU内存调整批处理大小

### 训练监控
- 每100步进行验证集评估
- 使用TensorBoard查看训练曲线
- 关注val_auc和val_logloss指标

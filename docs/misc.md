# 技术细节与扩展

## 性能优化

### 多进程数据加载优化

为解决数据I/O瓶颈（GPU使用率仅20%），项目实现了多进程分片数据加载：

#### 核心技术
- **ShardedDataset**: 基于文件分片的无重复数据加载
- **Worker级分片**: 每个进程处理不同的数据片段
- **统一接口**: 保持与原有DataLoader完全兼容

#### 性能提升
- **训练速度**: 从4.4 it/s提升到21.10 it/s（4-5x提升）
- **GPU利用率**: 大幅提升GPU使用效率
- **数据一致性**: 确保无数据重复或遗漏

#### 配置方式
```yaml
data:
  use_sharding: true    # 启用多进程分片
  num_workers: 8        # 数据加载进程数
  batch_size: 2048      # 优化批处理大小
```

#### 实现细节
```python
# 核心分片逻辑
def _get_worker_slice(self, worker_id: int, num_workers: int):
    total_lines = self._file_info['total_lines']
    lines_per_worker = total_lines // num_workers
    start_line = worker_id * lines_per_worker
    if worker_id == num_workers - 1:
        end_line = total_lines
    else:
        end_line = start_line + lines_per_worker
    return start_line, end_line
```

### 训练监控增强

#### 步级评估
- 每100步进行一次验证集评估
- 实时展示train_logloss、val_logloss、val_auc
- 步级进度百分比显示

#### 数据导出
- 训练指标自动保存为CSV: `training_metrics.csv`
- 测试结果自动保存为CSV: `test_results.csv`
- 所有数值保持4位小数精度

#### 示例输出
```
Step 100/220 (45.5%): train_logloss=0.5748, val_logloss=0.5363, val_auc=0.7267
Step 200/220 (90.9%): train_logloss=0.5520, val_logloss=0.5207, val_auc=0.7416
Step 220/220 (100.0%): train_logloss=0.5813, val_logloss=0.5283, val_auc=0.7424
```

## 特征ID化架构

本项目采用统一的特征ID化处理，提供更好的数据抽象：

### 设计理念
- **统一处理**: 所有特征（数值+类别）都经过相同的ID化→Hash→Embedding流程
- **Field感知**: 数值特征携带field信息，避免不同字段的相同值冲突
- **可扩展性**: 新数据集只需实现对应的ID化逻辑

### 核心组件
- `FeatureIDProcessor`: 统一特征ID化处理器
- `CriteoDataset`: 实现Criteo特定的ID化逻辑
- `EmbeddingMLP`: 适配统一embedding的模型架构

### ID化示例
```python
# 数值特征: field1的值10 → "1_10" → hash → embedding
# 类别特征: field14的值abc → "14_abc" → hash → embedding
```

### 架构优势

#### 统一抽象
所有特征都通过相同的处理流程：
```
原始特征 → 特征ID → Hash索引 → Embedding查找 → 特征向量
```

#### Field感知处理
不同字段的相同值会被区分对待：
- 用户年龄字段的"25" → "age_25"
- 商品价格字段的"25" → "price_25"

#### 扩展性强
添加新数据集只需实现ID化逻辑：
```python
class NewDataset(BaseCTRDataset):
    def _create_feature_id(self, field_idx, value):
        # 实现数据集特定的ID化逻辑
        return f"{field_idx}_{value}"
```

### 实现详情

#### FeatureIDProcessor类
```python
class FeatureIDProcessor:
    def __init__(self, hash_size: int = 1000000):
        self.hash_size = hash_size

    def process_features(self, features):
        """将原始特征转换为embedding索引"""
        ids = []
        for field_idx, value in enumerate(features):
            feature_id = self._create_feature_id(field_idx, value)
            hash_idx = self._hash_feature_id(feature_id)
            ids.append(hash_idx)
        return ids
```

#### Hash冲突处理
- 使用Python内置hash函数
- 可配置hash_size (默认100万)
- 冲突概率低，对模型影响有限

#### 内存优化
- 特征ID到hash的映射不需要存储
- 动态计算，节省内存开销
- 支持大规模稀疏特征处理

## 扩展开发

### 添加新模型

1. **创建模型类**
```python
# models/new_model.py
import torch.nn as nn

class NewCTRModel(nn.Module):
    def __init__(self, num_features, embedding_dim, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(1000000, embedding_dim)
        # 定义模型结构

    def forward(self, x):
        # 实现前向传播
        pass
```

2. **注册模型工厂**
```python
# scripts/train.py 中的 create_model 函数
def create_model(config, num_features):
    model_name = config['model']['name'].lower()
    if model_name == 'new_model':
        return NewCTRModel(num_features, **config['model'])
    # 其他模型...
```

3. **创建配置文件**
```yaml
# configs/new_model/new_model.yaml
model:
  name: "new_model"
  embedding_dim: 16
  # 模型特定参数
```

### 添加新数据集

1. **继承基类**
```python
# datasets/new_dataset.py
from .common import BaseCTRDataset

class NewDataset(BaseCTRDataset):
    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, **kwargs)
        self.num_features = 50  # 数据集特征数

    def _parse_line(self, line):
        # 实现数据集特定的解析逻辑
        pass

    def _create_feature_id(self, field_idx, value):
        # 实现特征ID化逻辑
        return f"{field_idx}_{value}"
```

2. **注册数据集工厂**
```python
# utils/data_loaders.py
def create_data_loaders(config):
    dataset_name = config['data']['dataset'].lower()
    if dataset_name == 'new_dataset':
        from datasets.new_dataset import NewDataset
        train_dataset = NewDataset(config['data']['train_path'])
    # 其他数据集...
```

### 添加新评估指标

1. **实现指标函数**
```python
# trainers/metrics.py
def precision_at_k(y_pred, y_true, k=100):
    """计算Precision@K指标"""
    # 实现具体计算逻辑
    pass

def recall_at_k(y_pred, y_true, k=100):
    """计算Recall@K指标"""
    # 实现具体计算逻辑
    pass
```

2. **注册到评估器**
```python
# trainers/ctr_trainer.py
class CTRTrainer:
    def _evaluate_metrics(self, y_pred, y_true, metrics):
        results = {}
        for metric in metrics:
            if metric == 'precision_at_k':
                results[metric] = precision_at_k(y_pred, y_true)
            elif metric == 'recall_at_k':
                results[metric] = recall_at_k(y_pred, y_true)
        return results
```

### 配置文件扩展

#### 模型配置模板
```yaml
model:
  name: "model_name"           # 模型标识
  embedding_dim: 16            # embedding维度
  # 模型特定参数

data:
  dataset: "dataset_name"      # 数据集标识
  batch_size: 1024            # 批处理大小
  num_workers: 4              # 数据加载进程数

training:
  epochs: 1                   # 训练轮数
  learning_rate: 0.001        # 学习率
  optimizer: "adam"           # 优化器

evaluation:
  metrics: ["auc", "logloss"] # 评估指标列表
```

#### 高级配置选项
```yaml
# 学习率调度
training:
  lr_scheduler:
    type: "StepLR"
    step_size: 10
    gamma: 0.1

# 早停策略
training:
  early_stopping:
    patience: 5
    metric: "val_auc"
    mode: "max"

# 模型保存策略
output:
  save_best: true              # 保存最佳模型
  save_last: true              # 保存最新模型
  save_interval: 1000          # 定期保存间隔
```

### 代码质量与测试

#### 单元测试
```python
# tests/test_models.py
import unittest
from models.embedding_mlp import EmbeddingMLP

class TestEmbeddingMLP(unittest.TestCase):
    def test_forward_pass(self):
        model = EmbeddingMLP(num_features=39, embedding_dim=8)
        # 测试前向传播
        pass
```

#### 代码规范
- 使用类型注解
- 遵循PEP8代码风格
- 添加详细的docstring文档
- 适当的异常处理

#### 性能测试
```python
# tests/test_performance.py
def test_training_speed():
    # 测试训练吞吐量
    pass

def test_memory_usage():
    # 测试内存占用
    pass
```

### 部署与生产

#### 模型导出
```python
# 导出为ONNX格式
torch.onnx.export(model, dummy_input, "model.onnx")

# 导出为TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

#### 推理服务
```python
# 构建Flask推理API
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model("path/to/checkpoint.pth")

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    prediction = model.predict(features)
    return jsonify({'prediction': prediction})
```

#### 性能监控
- 推理延迟监控
- 模型准确率监控
- 系统资源使用监控
- A/B测试框架集成
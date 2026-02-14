"""
模型分析工具：计算模型大小、参数量、FLOPs等指标
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Tuple, Any
from thop import profile, clever_format


def analyze_model_size(model: nn.Module, checkpoint_path: str = None) -> Dict[str, Any]:
    """
    分析模型大小，区分embedding和其他参数

    Args:
        model: PyTorch模型
        checkpoint_path: 模型checkpoint文件路径

    Returns:
        包含各种大小指标的字典
    """
    results = {}

    # 1. 参数统计 - 区分sparse和dense参数
    total_params = 0
    sparse_params = 0  # embedding table参数
    dense_params = 0   # 网络层参数

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if 'embedding' in name.lower():
            sparse_params += param_count
        else:
            dense_params += param_count

    results['parameters'] = {
        'total': total_params,
        'sparse_embedding': sparse_params,
        'dense_network': dense_params,
        'sparse_ratio': sparse_params / total_params if total_params > 0 else 0
    }

    # 2. 内存大小（参数）
    # 假设每个参数4字节（float32）
    bytes_per_param = 4
    results['memory_size'] = {
        'total_mb': total_params * bytes_per_param / (1024 * 1024),
        'embedding_mb': sparse_params * bytes_per_param / (1024 * 1024),
        'other_mb': dense_params * bytes_per_param / (1024 * 1024)
    }

    # 3. 文件大小（如果有checkpoint）
    if checkpoint_path and os.path.exists(checkpoint_path):
        file_size_bytes = os.path.getsize(checkpoint_path)
        results['file_size'] = {
            'total_mb': file_size_bytes / (1024 * 1024),
            'total_gb': file_size_bytes / (1024 * 1024 * 1024)
        }

    return results


def analyze_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    分析模型FLOPs

    Args:
        model: PyTorch模型
        input_shape: 输入张量形状 (batch_size, features)

    Returns:
        包含FLOPs指标的字典
    """
    results = {}

    # 创建示例输入，确保与模型在同一设备
    device = next(model.parameters()).device
    dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long, device=device)

    # 计算FLOPs
    model.eval()
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 格式化结果
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        per_example_flops = flops / input_shape[0] if input_shape[0] > 0 else flops

        results['flops'] = {
            'total': flops,
            'total_formatted': flops_formatted,
            'per_example': per_example_flops,
            'per_example_formatted': f"{per_example_flops:,.0f}"  # 直接格式化，不使用clever_format
        }

        results['thop_params'] = {
            'total': params,
            'total_formatted': params_formatted
        }

    except Exception as e:
        results['error'] = f"FLOPs计算失败: {str(e)}"

    return results


def print_model_analysis(model: nn.Module, checkpoint_path: str = None,
                        input_shape: Tuple[int, ...] = (1, 39)) -> None:
    """
    打印完整的模型分析报告

    Args:
        model: PyTorch模型
        checkpoint_path: checkpoint文件路径
        input_shape: 输入形状
    """
    print("=" * 60)
    print("模型分析报告")
    print("=" * 60)

    # 1. 模型大小分析
    size_analysis = analyze_model_size(model, checkpoint_path)

    print("\n📊 参数统计:")
    params = size_analysis['parameters']
    print(f"  总参数量: {params['total']:,}")
    print(f"  Embedding Table (Sparse)参数: {params['sparse_embedding']:,} ({params['sparse_ratio']:.2%})")
    print(f"  网络层(Dense)参数: {params['dense_network']:,} ({(1-params['sparse_ratio']):.2%})")

    print("\n💾 内存占用:")
    memory = size_analysis['memory_size']
    print(f"  总内存: {memory['total_mb']:.1f} MB")
    print(f"  Embedding内存: {memory['embedding_mb']:.1f} MB")
    print(f"  网络层内存: {memory['other_mb']:.1f} MB")

    if 'file_size' in size_analysis:
        print("\n📁 文件大小:")
        file_size = size_analysis['file_size']
        print(f"  Checkpoint大小: {file_size['total_mb']:.1f} MB ({file_size['total_gb']:.2f} GB)")

    # 2. FLOPs分析
    flops_analysis = analyze_model_flops(model, input_shape)

    if 'error' not in flops_analysis:
        print("\n⚡ 计算复杂度:")
        flops = flops_analysis['flops']
        print(f"  总FLOPs: {flops['total_formatted']}")
        print(f"  每样本FLOPs: {flops['per_example_formatted']}")

        # 计算推理速度估算（粗略）
        estimated_ops_per_sec = 1e12  # 假设1T OPS/sec
        estimated_samples_per_sec = estimated_ops_per_sec / flops['per_example']
        print(f"  估算推理速度: {estimated_samples_per_sec:,.0f} samples/sec")
    else:
        print(f"\n❌ FLOPs计算: {flops_analysis['error']}")

    print("\n" + "=" * 60)


def compare_models_efficiency(models_info: Dict[str, Dict]) -> None:
    """
    比较多个模型的效率指标

    Args:
        models_info: {model_name: {'model': model, 'checkpoint': path, 'metrics': {...}}}
    """
    print("=" * 80)
    print("模型效率对比")
    print("=" * 80)

    # 打印表头
    print(f"{'模型':<15} {'参数量':<12} {'文件大小':<12} {'FLOPs/样本':<15} {'AUC':<8}")
    print("-" * 75)

    for name, info in models_info.items():
        model = info['model']
        checkpoint = info.get('checkpoint')
        metrics = info.get('metrics', {})

        # 获取基本信息
        size_info = analyze_model_size(model, checkpoint)
        flops_info = analyze_model_flops(model, (1, 39))

        params = f"{size_info['parameters']['total']/1e6:.0f}M"
        file_size = f"{size_info.get('file_size', {}).get('total_mb', 0):.0f}MB"
        flops = flops_info.get('flops', {}).get('per_example_formatted', 'N/A')
        auc = f"{metrics.get('auc', 0):.3f}"

        print(f"{name:<15} {params:<12} {file_size:<12} {flops:<15} {auc:<8}")

    print("=" * 80)
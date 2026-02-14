#!/usr/bin/env python3
"""
Script: prepare_taobaoads.py
功能: 将 Taobao Ads 数据集的三个文件拼接并划分为 train (90%) + val (1%) + eval (9%)
处理流程:
1. 拼接 raw_sample.csv, ad_feature.csv, user_profile.csv
2. 只保留 clk 作为 label（移除 nonclk）
3. 移除 header -> shuffle -> split (无 header 输出)

用法: python scripts/prepare_taobaoads.py data/raw/taobaoads data/raw/taobaoads
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/prepare_taobaoads.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Taobao Ads Data Preparation Script")
    print("=" * 60)

    # 文件路径
    raw_sample_path = os.path.join(input_dir, "raw_sample.csv")
    ad_feature_path = os.path.join(input_dir, "ad_feature.csv")
    user_profile_path = os.path.join(input_dir, "user_profile.csv")

    # Step 1: 读取文件
    print("\n[Step 1/6] 读取数据文件...")
    print(f"  - 读取 raw_sample.csv...")
    raw_sample = pd.read_csv(raw_sample_path)
    print(f"    Shape: {raw_sample.shape}")
    print(f"    Columns: {list(raw_sample.columns)}")

    print(f"  - 读取 ad_feature.csv...")
    ad_feature = pd.read_csv(ad_feature_path)
    print(f"    Shape: {ad_feature.shape}")
    print(f"    Columns: {list(ad_feature.columns)}")

    print(f"  - 读取 user_profile.csv...")
    user_profile = pd.read_csv(user_profile_path)
    print(f"    Shape: {user_profile.shape}")
    print(f"    Columns: {list(user_profile.columns)}")

    # Step 2: 拼接数据
    print("\n[Step 2/6] 拼接数据...")
    print("  - 拼接 raw_sample 和 ad_feature (key: adgroup_id)...")
    merged = pd.merge(raw_sample, ad_feature, on='adgroup_id', how='left')
    print(f"    Shape after merge: {merged.shape}")

    print("  - 拼接 merged 和 user_profile (key: user/userid)...")
    # 重命名 raw_sample 的 user 列为 userid，以便拼接
    merged = merged.rename(columns={'user': 'userid'})
    merged = pd.merge(merged, user_profile, on='userid', how='left')
    print(f"    Shape after merge: {merged.shape}")

    # Step 3: 只保留 clk 作为 label，移除 nonclk
    print("\n[Step 3/6] 处理 label...")
    print(f"  - 移除 nonclk 列，只保留 clk 作为 label")
    if 'nonclk' in merged.columns:
        merged = merged.drop(columns=['nonclk'])

    # 将 clk 移到第一列作为 label
    cols = merged.columns.tolist()
    cols.remove('clk')
    merged = merged[['clk'] + cols]
    print(f"    Final shape: {merged.shape}")
    print(f"    Final columns: {list(merged.columns)}")

    # Step 3.5: 修正数据类型（整数字段不应该变成浮点数）
    print("\n  - 修正数据类型...")
    # 整数列（包括有缺失值的）
    int_columns = ['clk', 'userid', 'time_stamp', 'adgroup_id', 'cate_id', 'campaign_id',
                   'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code',
                   'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']

    # pid 是字符串，price 是浮点数，其他都是整数
    # 对于有缺失值的整数列，填充为 0
    for col in int_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # price 是浮点数
    if 'price' in merged.columns:
        merged['price'] = merged['price'].fillna(0.0).astype(float)

    # pid 是字符串
    if 'pid' in merged.columns:
        merged['pid'] = merged['pid'].fillna('').astype(str)

    print(f"    Data types after correction:")
    print(merged.dtypes)

    # 统计正样本率
    total_samples = len(merged)
    positive_samples = merged['clk'].sum()
    positive_rate = positive_samples / total_samples * 100
    print(f"\n  - 正样本率: {positive_rate:.4f}% ({positive_samples}/{total_samples})")

    # Step 4: Shuffle
    print("\n[Step 4/6] Shuffle 数据...")
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    print("  - Shuffle 完成")

    # Step 5: Split
    print("\n[Step 5/6] 切分数据 (90% train / 1% val / 9% eval)...")
    train_ratio = 0.90
    val_ratio = 0.01

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_data = merged.iloc[:train_size]
    val_data = merged.iloc[train_size:train_size + val_size]
    eval_data = merged.iloc[train_size + val_size:]

    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val:   {len(val_data)} samples")
    print(f"  - Eval:  {len(eval_data)} samples")

    # Step 6: 保存（无 header，用逗号分隔）
    print("\n[Step 6/6] 保存数据...")
    train_path = os.path.join(output_dir, "train_split.txt")
    val_path = os.path.join(output_dir, "val_split.txt")
    eval_path = os.path.join(output_dir, "eval_split.txt")

    print(f"  - 保存 train_split.txt...")
    train_data.to_csv(train_path, index=False, header=False)

    print(f"  - 保存 val_split.txt...")
    val_data.to_csv(val_path, index=False, header=False)

    print(f"  - 保存 eval_split.txt...")
    eval_data.to_csv(eval_path, index=False, header=False)

    # 验证
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"  训练集: {train_path} ({len(train_data)} 行)")
    print(f"  验证集: {val_path} ({len(val_data)} 行)")
    print(f"  评估集: {eval_path} ({len(eval_data)} 行)")
    print(f"\n  总行数: {total_samples}")
    print(f"  正样本率: {positive_rate:.4f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/bin/bash
# Script: prepare_criteo.sh
# 功能: 将 Criteo Kaggle 版 train.txt 划分为 train (90%) + val (1%) + eval (9%)
# 注意: 当前版本关闭了shuffle功能，按原始数据顺序切分
# 用法: bash scripts/prepare_criteo.sh data/raw/criteo_mini/train.txt data/raw/criteo_mini

# set -euo pipefail  # 临时关闭严格模式用于调试

INPUT_FILE=$1       # 原始 train.txt 路径
OUTPUT_DIR=$2       # 输出目录，例如 data/raw/criteo_mini

mkdir -p "${OUTPUT_DIR}"

# 行数统计
TOTAL=$(wc -l < "${INPUT_FILE}")
TRAIN_LINES=$(( TOTAL * 90 / 100 ))
VAL_LINES=$(( TOTAL * 1 / 100 ))
EVAL_LINES=$(( TOTAL - TRAIN_LINES - VAL_LINES ))

echo "[Info] 总行数: $TOTAL"
echo "[Info] 训练集: $TRAIN_LINES 行 (90%)"
echo "[Info] 验证集: $VAL_LINES 行 (1%)"
echo "[Info] 评估集: $EVAL_LINES 行 (9%)"

# 暂时跳过shuffle，直接使用原始文件顺序
echo "[Info] 跳过shuffle，使用原始数据顺序进行切分..."

# 三段式切分（基于原始数据）
head -n ${TRAIN_LINES} "${INPUT_FILE}" > "${OUTPUT_DIR}/train_split.txt"

# 跳过训练集，提取验证集
tail -n +$(( TRAIN_LINES + 1 )) "${INPUT_FILE}" | head -n ${VAL_LINES} > "${OUTPUT_DIR}/val_split.txt"

# 跳过训练集和验证集，提取评估集
tail -n ${EVAL_LINES} "${INPUT_FILE}" > "${OUTPUT_DIR}/eval_split.txt"

echo "[Info] 数据切分完成（未shuffle）"

# 验证
echo "[Done] 三段式划分完成:"
wc -l "${OUTPUT_DIR}/train_split.txt" "${OUTPUT_DIR}/val_split.txt" "${OUTPUT_DIR}/eval_split.txt"

# 验证总数
TOTAL_CHECK=$(( $(wc -l < "${OUTPUT_DIR}/train_split.txt") + $(wc -l < "${OUTPUT_DIR}/val_split.txt") + $(wc -l < "${OUTPUT_DIR}/eval_split.txt") ))
echo "[Check] 总行数验证: 原始=$TOTAL, 切分后=$TOTAL_CHECK"

# 统计正样本率（优化版本）
echo ""
echo "[Stats] 正样本率统计:"

# 使用更快的方法统计正样本率
echo "  原始数据: 正样本率计算中..."
ORIG_TOTAL=$(wc -l < "${INPUT_FILE}")
ORIG_POSITIVE=$(cut -f1 "${INPUT_FILE}" | grep -c "^1$")
ORIG_RATE=$(echo "scale=4; $ORIG_POSITIVE * 100 / $ORIG_TOTAL" | bc -l)
echo "  原始数据: 正样本率 = ${ORIG_RATE}% ($ORIG_POSITIVE/$ORIG_TOTAL)"

echo "  训练集:   正样本率计算中..."
TRAIN_TOTAL=$(wc -l < "${OUTPUT_DIR}/train_split.txt")
TRAIN_POSITIVE=$(cut -f1 "${OUTPUT_DIR}/train_split.txt" | grep -c "^1$")
TRAIN_RATE=$(echo "scale=4; $TRAIN_POSITIVE * 100 / $TRAIN_TOTAL" | bc -l)
echo "  训练集:   正样本率 = ${TRAIN_RATE}% ($TRAIN_POSITIVE/$TRAIN_TOTAL)"

echo "  验证集:   正样本率计算中..."
VAL_TOTAL=$(wc -l < "${OUTPUT_DIR}/val_split.txt")
VAL_POSITIVE=$(cut -f1 "${OUTPUT_DIR}/val_split.txt" | grep -c "^1$")
VAL_RATE=$(echo "scale=4; $VAL_POSITIVE * 100 / $VAL_TOTAL" | bc -l)
echo "  验证集:   正样本率 = ${VAL_RATE}% ($VAL_POSITIVE/$VAL_TOTAL)"

echo "  评估集:   正样本率计算中..."
EVAL_TOTAL=$(wc -l < "${OUTPUT_DIR}/eval_split.txt")
EVAL_POSITIVE=$(cut -f1 "${OUTPUT_DIR}/eval_split.txt" | grep -c "^1$")
EVAL_RATE=$(echo "scale=4; $EVAL_POSITIVE * 100 / $EVAL_TOTAL" | bc -l)
echo "  评估集:   正样本率 = ${EVAL_RATE}% ($EVAL_POSITIVE/$EVAL_TOTAL)"
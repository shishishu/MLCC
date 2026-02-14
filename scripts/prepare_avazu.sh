#!/bin/bash
# Script: prepare_avazu.sh
# 功能: 将 Avazu train.csv 划分为 train (90%) + val (1%) + eval (9%)
# 处理流程: 移除header -> shuffle -> split (无header输出)
# 用法: bash scripts/prepare_avazu.sh data/raw/avazu/train.csv data/raw/avazu

set -euo pipefail

INPUT_FILE=$1       # 原始 train.csv 路径
OUTPUT_DIR=$2       # 输出目录，例如 data/raw/avazu

mkdir -p "${OUTPUT_DIR}"

# 行数统计 (排除header)
TOTAL_WITH_HEADER=$(wc -l < "${INPUT_FILE}")
TOTAL=$(( TOTAL_WITH_HEADER - 1 ))  # 去除header行
TRAIN_LINES=$(( TOTAL * 90 / 100 ))
VAL_LINES=$(( TOTAL * 1 / 100 ))
EVAL_LINES=$(( TOTAL - TRAIN_LINES - VAL_LINES ))

echo "[Info] 总行数 (不含header): $TOTAL"
echo "[Info] 训练集: $TRAIN_LINES 行 (90%)"
echo "[Info] 验证集: $VAL_LINES 行 (1%)"
echo "[Info] 评估集: $EVAL_LINES 行 (9%)"

# 移除header并shuffle
echo "[Info] 移除header并shuffle数据..."
SHUFFLED_FILE="${OUTPUT_DIR}/temp_shuffled.txt"
tail -n +2 "${INPUT_FILE}" | shuf > "${SHUFFLED_FILE}"

# 统计原始数据正样本率（用于对比）
echo ""
echo "[Stats] 原始数据统计:"
ORIG_TOTAL=$TOTAL
ORIG_POSITIVE=$(tail -n +2 "${INPUT_FILE}" | cut -d',' -f2 | grep -c "^1$")
ORIG_RATE=$(echo "scale=4; $ORIG_POSITIVE * 100 / $ORIG_TOTAL" | bc -l)
echo "  正样本率 = ${ORIG_RATE}% ($ORIG_POSITIVE/$ORIG_TOTAL)"

# 三段式切分（无header输出）- 使用一次遍历优化性能
echo ""
echo "[Info] 开始三段式切分（一次遍历）..."

# 使用 awk 一次遍历文件，同时输出到三个文件
awk -v train_lines=${TRAIN_LINES} \
    -v val_lines=${VAL_LINES} \
    -v train_file="${OUTPUT_DIR}/train_split.txt" \
    -v val_file="${OUTPUT_DIR}/val_split.txt" \
    -v eval_file="${OUTPUT_DIR}/eval_split.txt" '
    NR <= train_lines {
        print > train_file
        next
    }
    NR <= train_lines + val_lines {
        print > val_file
        next
    }
    {
        print > eval_file
    }
' "${SHUFFLED_FILE}"

echo "  文件切分完成，正在统计..."

# 1. 训练集统计
echo "  [1/3] 统计训练集..."
TRAIN_COUNT=$(wc -l < "${OUTPUT_DIR}/train_split.txt")
TRAIN_POSITIVE=$(cut -d',' -f2 "${OUTPUT_DIR}/train_split.txt" | grep -c "^1$")
TRAIN_RATE=$(echo "scale=4; $TRAIN_POSITIVE * 100 / $TRAIN_COUNT" | bc -l)
echo "        ${OUTPUT_DIR}/train_split.txt: $TRAIN_COUNT 行, 正样本率 = ${TRAIN_RATE}% ($TRAIN_POSITIVE/$TRAIN_COUNT)"

# 2. 验证集统计
echo "  [2/3] 统计验证集..."
VAL_COUNT=$(wc -l < "${OUTPUT_DIR}/val_split.txt")
VAL_POSITIVE=$(cut -d',' -f2 "${OUTPUT_DIR}/val_split.txt" | grep -c "^1$")
VAL_RATE=$(echo "scale=4; $VAL_POSITIVE * 100 / $VAL_COUNT" | bc -l)
echo "        ${OUTPUT_DIR}/val_split.txt: $VAL_COUNT 行, 正样本率 = ${VAL_RATE}% ($VAL_POSITIVE/$VAL_COUNT)"

# 3. 评估集统计
echo "  [3/3] 统计评估集..."
EVAL_COUNT=$(wc -l < "${OUTPUT_DIR}/eval_split.txt")
EVAL_POSITIVE=$(cut -d',' -f2 "${OUTPUT_DIR}/eval_split.txt" | grep -c "^1$")
EVAL_RATE=$(echo "scale=4; $EVAL_POSITIVE * 100 / $EVAL_COUNT" | bc -l)
echo "        ${OUTPUT_DIR}/eval_split.txt: $EVAL_COUNT 行, 正样本率 = ${EVAL_RATE}% ($EVAL_POSITIVE/$EVAL_COUNT)"

# 清理临时文件
rm "${SHUFFLED_FILE}"

# 最终验证
echo ""
echo "[Done] 数据切分完成（已shuffle，无header）"
TOTAL_CHECK=$(( TRAIN_COUNT + VAL_COUNT + EVAL_COUNT ))
echo "[Check] 总行数验证: 原始=$TOTAL, 切分后=$TOTAL_CHECK"

# 汇总统计
echo ""
echo "========================================"
echo "数据集划分汇总:"
echo "----------------------------------------"
echo "原始数据: $ORIG_TOTAL 行, 正样本率 = ${ORIG_RATE}%"
echo "训练集:   $TRAIN_COUNT 行 ($(echo "scale=2; $TRAIN_COUNT * 100 / $TOTAL" | bc -l)%), 正样本率 = ${TRAIN_RATE}%"
echo "验证集:   $VAL_COUNT 行 ($(echo "scale=2; $VAL_COUNT * 100 / $TOTAL" | bc -l)%), 正样本率 = ${VAL_RATE}%"
echo "评估集:   $EVAL_COUNT 行 ($(echo "scale=2; $EVAL_COUNT * 100 / $TOTAL" | bc -l)%), 正样本率 = ${EVAL_RATE}%"
echo "========================================"

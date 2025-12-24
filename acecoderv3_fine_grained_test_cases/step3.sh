#!/bin/bash
set -e  # 出错立即退出
set -o pipefail

# -------------------------
# 配置参数
# -------------------------
NUM_ROUNDS=${1:-4}             # 默认循环4轮，可通过命令行指定，如 ./run_test_generation.sh 3
NUM_PROC=64                    # 并行进程数
MODEL_NAME="gpt-4.1-mini"      # 使用的模型
SAVE_BATCH_SIZE=8
MAX_CONCURRENT=64
BATCH_DELAY=0.01
BASE_DIR="acecoderv3_fine_grained_test_cases/outputs/remaining"

# -------------------------
# 第1轮初始输入文件（来自 step2.2）
# -------------------------
INPUT_FILE="${BASE_DIR}/step2.2_merged_eval.jsonl"

# -------------------------
# 主循环
# -------------------------
for ROUND in $(seq 1 $NUM_ROUNDS); do
    echo "=============================="
    echo "🚀 开始第 ${ROUND} 轮 Test Generation"
    echo "=============================="

    echo "👉 Step 3.1: Filter tests"
    python acecoderv3_fine_grained_test_cases/step3.1_filter_tests.py \
        "${INPUT_FILE}" --round "${ROUND}" --num_proc "${NUM_PROC}"

    echo "👉 Step 3.2: Generate tests"
    python acecoderv3_fine_grained_test_cases/step3.2_gen_tests.py \
        "${BASE_DIR}/round_${ROUND}/step3.1_filter_tests_round_${ROUND}.jsonl" \
        --round "${ROUND}" \
        --model_name "${MODEL_NAME}" \
        --save_batch_size "${SAVE_BATCH_SIZE}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --batch_delay "${BATCH_DELAY}"

    echo "👉 Step 3.3: Parse generated tests"
    python acecoderv3_fine_grained_test_cases/step3.3_parsing_tests.py \
        "${BASE_DIR}/round_${ROUND}/step3.2_gen_tests_results_round_${ROUND}.jsonl" \
        --round "${ROUND}" --num_proc "${NUM_PROC}"

    # echo "👉 Step 3.4: Evaluate combined tests"
    # python acecoderv3_fine_grained_test_cases/step3.4_eval_combined_tests.py \
    #     "${BASE_DIR}/round_${ROUND}/step3.3_parsing_tests_round_${ROUND}.jsonl" \
    #     --round "${ROUND}" --num_proc "${NUM_PROC}"

    echo "👉 Step 3.4: Filter evaluated tests"
    python acecoderv3_fine_grained_test_cases/step3.4_filter_tests.py \
        "${BASE_DIR}/round_${ROUND}/step3.3_parsing_tests_round_${ROUND}.jsonl" \
        --num_proc "${NUM_PROC}" --round "${ROUND}"

    echo "👉 Step 3.5: Generate refined tests"
    python acecoderv3_fine_grained_test_cases/step3.5_gen_tests.py \
        "${BASE_DIR}/round_${ROUND}/step3.4_filter_tests_round_${ROUND}.jsonl" \
        --round "${ROUND}" \
        --model_name "${MODEL_NAME}" \
        --save_batch_size "${SAVE_BATCH_SIZE}" \
        --max_concurrent 1 \
        --batch_delay "${BATCH_DELAY}"

    echo "👉 Step 3.6: Parse refined tests"
    python acecoderv3_fine_grained_test_cases/step3.6_parsing_tests.py \
        "${BASE_DIR}/round_${ROUND}/step3.5_gen_tests_results_round_${ROUND}.jsonl" \
        --round "${ROUND}" --num_proc "${NUM_PROC}"

    # echo "👉 Step 3.7: Evaluate final tests"
    # python acecoderv3_fine_grained_test_cases/step3.7_eval.py \
    #     "${BASE_DIR}/round_${ROUND}/step3.6_parsing_tests_round_${ROUND}.jsonl" \
    #     --round "${ROUND}" --num_proc "${NUM_PROC}"

    # 为下一轮准备输入文件
    INPUT_FILE="${BASE_DIR}/round_${ROUND}/step3.7_eval_round_${ROUND}.jsonl"

    echo "✅ 第 ${ROUND} 轮完成！结果保存在：${INPUT_FILE}"
    echo
done

echo "🎉 所有 ${NUM_ROUNDS} 轮 test generation 完成！"

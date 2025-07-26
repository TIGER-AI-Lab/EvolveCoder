#!/bin/bash
set -e
# Parameters
DATASET_NAME="ise-uiuc/Magicoder-Evol-Instruct-110K"
MAX_SAMPLES=100 # set to 0 for all samples
MODEL_NAME="o4-mini"
SAVE_BATCH_SIZE=25
MAX_CONCURRENT=25
# GEN_MODEL="Qwen/Qwen3-8B"
GEN_MODEL="gpt-4.1-mini"
USE_VLLM=False
TENSOR_PARALLEL_SIZE=4
TOP_P=0.95
TOP_K=1
TEMPERATURE=0.6
MAX_TOKENS=32768
N=8
SEED=42
OVERWRITE=False

# Pretty name function equivalent in bash
pretty_name() {
    echo "$1" | sed 's/.*\///g' | sed 's/-/_/g'
}

# Auto-inferred file paths
DATASET_DIR=$(pretty_name "$DATASET_NAME")
MODEL_DIR=$(pretty_name "$MODEL_NAME")
GEN_MODEL_SHORT=$(pretty_name "$GEN_MODEL")


STEP1_OUTPUT="outputs/${DATASET_DIR}/${MODEL_DIR}/step1_prompting_results.jsonl"
STEP1_1_OUTPUT="outputs/${DATASET_DIR}/${MODEL_DIR}/step1.1_parsing.jsonl"
STEP2_1_OUTPUT="outputs/${DATASET_DIR}/${MODEL_DIR}/step2.1_gen_${GEN_MODEL_SHORT}_seed${SEED}.jsonl"
STEP2_2_OUTPUT="outputs/${DATASET_DIR}/${MODEL_DIR}/step2.2_eval_${GEN_MODEL_SHORT}_seed${SEED}.jsonl"
echo "Step 1 Output: $STEP1_OUTPUT"
echo "Step 1.1 Output: $STEP1_1_OUTPUT"
echo "Step 2.1 Output: $STEP2_1_OUTPUT"
echo "Step 2.2 Output: $STEP2_2_OUTPUT"

# Run pipeline
python step1_prompting.py --dataset_name $DATASET_NAME --max_samples $MAX_SAMPLES --model_name $MODEL_NAME --save_batch_size $SAVE_BATCH_SIZE --max_concurrent $MAX_CONCURRENT --overwrite $OVERWRITE

python step1.1_parsing.py --file_path $STEP1_OUTPUT

# use vllm or openai for generation
if [ "$USE_VLLM" = true ]; then
    python step2.1_vllm_gen.py $STEP1_1_OUTPUT \
        --model_name_or_path=$GEN_MODEL \
        --overwrite $OVERWRITE \
        --save_batch_size=$SAVE_BATCH_SIZE \
        --tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
        --top_p=$TOP_P --top_k=$TOP_K --temperature=$TEMPERATURE --max_tokens=$MAX_TOKENS --n $N
else
    python step2.1_openai_gen.py $STEP1_1_OUTPUT \
        --model=$GEN_MODEL \
        --overwrite $OVERWRITE \
        --batch_size=$SAVE_BATCH_SIZE \
        --max_concurrent=$MAX_CONCURRENT \
        --top_p=$TOP_P --temperature=$TEMPERATURE --max_tokens=$MAX_TOKENS --n $N
fi

python step2.2_eval.py $STEP2_1_OUTPUT --overwrite $OVERWRITE

python step_3.1_filter_tests.py $STEP2_2_OUTPUT --overwrite $OVERWRITE
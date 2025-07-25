#!/bin/bash
set -e
# Parameters
DATASET_NAME="ise-uiuc/Magicoder-Evol-Instruct-110K"
MAX_SAMPLES=20 # set to 0 for all samples
MODEL_NAME="o4-mini"
SAVE_BATCH_SIZE=5
MAX_CONCURRENT=25
GEN_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
TENSOR_PARALLEL_SIZE=2
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
echo "Step 1 Output: $STEP1_OUTPUT"
echo "Step 1.1 Output: $STEP1_1_OUTPUT"
echo "Step 2.1 Output: $STEP2_1_OUTPUT"


# Run pipeline
python step1_prompting.py --dataset_name $DATASET_NAME --max_samples $MAX_SAMPLES --model_name $MODEL_NAME --save_batch_size $SAVE_BATCH_SIZE --max_concurrent $MAX_CONCURRENT --overwrite $OVERWRITE

python step1.1_parsing.py --file_path $STEP1_OUTPUT

python step2.1_vllm_gen.py $STEP1_1_OUTPUT \
    --save_batch_size=$SAVE_BATCH_SIZE \
    --model_name_or_path=$GEN_MODEL \
    --tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
    --overwrite $OVERWRITE \
    --top_p=$TOP_P --top_k=$TOP_K --temperature=$TEMPERATURE --max_tokens=$MAX_TOKENS --n $N

python step2.2_eval.py $STEP2_1_OUTPUT --overwrite $OVERWRITE
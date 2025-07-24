#!/bin/bash
DATA=outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.json
model_name_or_path='Qwen/Qwen2.5-Coder-7B-Instruct'

#######################################
TOTAL=$( wc -l < "${DATA}" )
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
BSZ=$(( TOTAL / GPUS + 1 ))
max_tokens=32768
top_p=0.95
temperature=0.6
tensor_parallel_size=1
echo "Total: ${TOTAL}, GPUs: ${GPUS}, Batch Size: ${BSZ}, Model: ${model_name_or_path}"
#######################################

chunk=$(( (TOTAL + GPUS - 1) / GPUS ))

for (( gpu=0; gpu<GPUS; gpu++ )); do
    start=$(( gpu * chunk ))
    end=$(( start + chunk ))
    (( end > TOTAL )) && end=$TOTAL
    (( start >= TOTAL )) && break

    echo "Processing GPU $gpu: start=$start, end=$end"
    
    echo "parameters:"
    echo "  DATA: ${DATA}"
    echo "  start_idx: ${start}"
    echo "  end_idx: ${end}"
    echo "  batch_size: ${BSZ}"
    echo "  model_name_or_path: ${model_name_or_path}"
    echo "  max_tokens: ${max_tokens}"
    echo "  top_p: ${top_p}"

    python step2.1_vllm_gen.py "${DATA}" \
        --start_idx="${start}" \
        --end_idx="${end}" \
        --save_batch_size="${BSZ}" \
        --model_name_or_path="${model_name_or_path}" \
        --tensor_parallel_size=${tensor_parallel_size} \
        --top_p="${top_p}" \
        --temperature="${temperature}" \
        --max_tokens="${max_tokens}" \
        --device_id="${gpu}" &
done

wait
echo "All GPUs finished."







# High performance settings with aiohttp
python step2.1_openai_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.json \
    --start_idx=0 \
    --end_idx=50 \
    --batch_size=25 \
    --max_concurrent=25 \
    --model='gpt-4.1-mini' \
    --top_p=0.95 \
    --temperature=0.6 \
    --max_tokens=4000


python step2.1_vllm_gen.py outputs/step1.1_parsing/Magicoder_Evol_Instruct_110K_gpt_4o_mini.json \
    --start_idx=0 \
    --end_idx=50 \
    --save_batch_size=16 \
    --model_name_or_path='Qwen/Qwen2.5-Coder-7B-Instruct' \
    --tensor_parallel_size=2 \
    --top_p=0.95 --top_k=1 --temperature=0.6 --max_tokens=2048


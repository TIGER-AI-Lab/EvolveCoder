#!/bin/bash
DATA=acecoderv3/outputs/all_20/gpt_4.1_mini/step1.1_parsing.jsonl
model_name_or_path='Qwen/Qwen3-4B'
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#######################################
TOTAL=$( wc -l < "${DATA}" )

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    GPUS=${#GPU_ARRAY[@]}
else
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    GPU_ARRAY=()
    for (( i=0; i<GPUS; i++ )); do
        GPU_ARRAY+=($i)
    done
fi

max_tokens=32768
top_p=0.95
top_k=20
temperature=0.6
n=16
tensor_parallel_size=1

if (( GPUS % tensor_parallel_size != 0 )); then
    echo "Error: Number of GPUs ($GPUS) must be divisible by tensor_parallel_size ($tensor_parallel_size)"
    exit 1
fi

NUM_PROCESSES=$(( GPUS / tensor_parallel_size ))
BSZ=$(( TOTAL / NUM_PROCESSES + 1 ))

echo "Total: ${TOTAL}, GPUs: ${GPUS}, Tensor Parallel Size: ${tensor_parallel_size}"
echo "Number of Processes: ${NUM_PROCESSES}, Batch Size: ${BSZ}, Model: ${model_name_or_path}"
echo "GPU IDs: ${GPU_ARRAY[@]}"
#######################################

chunk=$(( (TOTAL + NUM_PROCESSES - 1) / NUM_PROCESSES ))

for (( proc_idx=0; proc_idx<NUM_PROCESSES; proc_idx++ )); do
    gpu_start=$(( proc_idx * tensor_parallel_size ))
    gpu_end=$(( gpu_start + tensor_parallel_size ))
    
    process_gpus=()
    for (( g=gpu_start; g<gpu_end; g++ )); do
        process_gpus+=(${GPU_ARRAY[$g]})
    done
    
    gpu_list=$(IFS=','; echo "${process_gpus[*]}")
    
    start=$(( proc_idx * chunk ))
    end=$(( start + chunk ))
    (( end > TOTAL )) && end=$TOTAL
    (( start >= TOTAL )) && break

    echo "=========================================="
    echo "Process $proc_idx: GPUs [$gpu_list], data range [$start, $end)"
    echo "=========================================="
    
    echo "parameters:"
    echo "  DATA: ${DATA}"
    echo "  start_idx: ${start}"
    echo "  end_idx: ${end}"
    echo "  batch_size: ${BSZ}"
    echo "  model_name_or_path: ${model_name_or_path}"
    echo "  tensor_parallel_size: ${tensor_parallel_size}"
    echo "  max_tokens: ${max_tokens}"
    echo "  top_p: ${top_p}"
    echo ""

    CUDA_VISIBLE_DEVICES=$gpu_list python acecoderv3/step2.1_vllm_gen.py "${DATA}" \
        --start_idx="${start}" \
        --end_idx="${end}" \
        --save_batch_size="${BSZ}" \
        --model_name_or_path="${model_name_or_path}" \
        --tensor_parallel_size=${tensor_parallel_size} \
        --top_p="${top_p}" \
        --top_k="${top_k}" \
        --temperature="${temperature}" \
        --n="${n}" \
        --max_tokens="${max_tokens}" &
done

wait
echo "All processes finished."




#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export OMP_NUM_THREADS=1
SWANLAB_MODE_VALUE="${SWANLAB_MODE:-cloud}"

MODEL_DIR=./Qwen/Qwen3-1.7B-Base
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "Missing base model at ${MODEL_DIR}. Download Qwen3-1.7B-Base there or update MODEL_DIR in sft.sh." >&2
    exit 1
fi

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    sid_index_path=./data/Amazon/index/${category}.index.json
    item_meta_path=./data/Amazon/index/${category}.item.json
    output_dir=./output/qwen3-1.7b-base-${category}-sft
    swanlab_run_name=qwen3-1.7b-base-${category}-sft
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 4 \
            sft.py \
            --base_model ${MODEL_DIR} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ${output_dir} \
            --swanlab_project MiniOneRec \
            --swanlab_run_name ${swanlab_run_name} \
            --swanlab_mode ${SWANLAB_MODE_VALUE} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ${sid_index_path} \
            --item_meta_path ${item_meta_path} \
            --freeze_LLM False
done

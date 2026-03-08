#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export OMP_NUM_THREADS=1
SWANLAB_MODE_VALUE="${SWANLAB_MODE:-cloud}"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    sid_index_path=./data/Amazon/index/${category}.index.json
    item_meta_path=./data/Amazon/index/${category}.item.json
    model_path=./output/qwen3-1.7b-base-${category}-sft
    output_dir=./output/qwen3-1.7b-base-${category}-rl
    swanlab_run_name=qwen3-1.7b-base-${category}-rl
    eval_step=0.5
    save_step=0.5
    save_total_limit=3

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 4 --main_process_port 29503 \
                                    rl.py \
                        --model_path ${model_path} \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step ${eval_step} \
                        --save_step ${save_step} \
                        --save_total_limit ${save_total_limit} \
                        --keep_best_checkpoint True \
                        --best_metric eval_reward \
                        --greater_is_better True \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ${output_dir} \
                        --swanlab_project MiniOneRec \
                        --swanlab_run_name ${swanlab_run_name} \
                        --swanlab_mode ${SWANLAB_MODE_VALUE} \
                        --sid_index_path ${sid_index_path} \
                        --item_meta_path ${item_meta_path}
done

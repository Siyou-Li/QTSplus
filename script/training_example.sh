#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export PROJECT_PATH=<your_project_path>/QTSplus
export CHECKPOINT_NAME=QTSplus-3B
export TOKENIZERS_PARALLELISM=True

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29502  \
    src/train/train.py \
    --version v0 \
    --pretrain_lm_model  $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-LM\
    --lm_model_type qwen2_5_vl_causal_lm \
    --lora_enable False \
    --vision_tower qwen2_5_vl_vision \
    --pretrain_vision_model $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-Vision/model.safetensors \
    --vision_processor $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-Vision \
    --bf16 True \
    --train_base_path datasets/ShareGPTVideoChoice/train_300k_480p \
    --train_jsonl_path $PROJECT_PATH/datasets/ShareGPTVideoChoice/3b/qa/prediction_correct_train.jsonl \
    --val_base_path datasets/ShareGPTVideoChoice/train_300k_480p \
    --val_jsonl_path $PROJECT_PATH/datasets/ShareGPTVideoChoice/3b/qa/prediction_correct_train.jsonl \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --dataset_type qa \
    --model_max_length 512 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --max_grad_norm 1 \
    --gradient_checkpointing True \
    --dataloader_pin_memory False \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --wandb_project_name QTS+ \
    --wandb_run_name $CHECKPOINT_NAME \
    --freeze_vision_model True \
    --freeze_lm True \
    --freeze_qts_scoring_layers False \
    --qts_plus_tau_s 0.5\
    --qts_plus_nmax 512\
    --qts_plus_rho_min 0.05\
    --qts_plus_rho_max 0.5\
    --qts_plus_block_dropout 0.0\
    --qts_plus_reencode True \
    --qts_plus_scoring_layers 1 \
    --qts_plus_reencode_layers 2 \
    --project_text_if_needed False\
    --lambda_t 0.1\
    --lambda_m 0.17\
    --lambda_s 0.05


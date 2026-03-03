#!/bin/bash
# Training recipe from:
# https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html
# https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_full.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Experiment
expr_name=mini_internvl_4b_video_ft_8-16-16
train_config=config_internvl
train_corpus=videochat2_video
output_dir=workspace/outputs/${expr_name}

# Model
repo_id=workspace/models/Mini-InternVL-Chat-4B-V1-5
num_frames=8

# Training scale
num_gpus=4
full_batch_size=128 # Keep full_batch_size == num_gpus * batch_size * gradient_accumulation_steps
batch_size=16
gradient_accumulation_steps=2
num_save_samples=100000

# Hyperparameters
epochs=3
lr=4e-5
min_lr=0
wd=0.05
warmup_ratio=0.03

# Compute steps
save_steps=$(($num_save_samples/($batch_size*$num_gpus)))
ckpt_steps=$(($save_steps/10))

# Python path
which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.

# Debug info
echo "PYTHONPATH: ${PYTHONPATH}"
echo "which python: ${which_python}"
echo "full_batch_size: ${full_batch_size}"
echo "batch_size: ${batch_size}"
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"
echo "save_steps: ${save_steps}"
echo "ckpt_steps: ${ckpt_steps}"

# Launch training
accelerate launch \
    --config_file scripts/train/accel_config_deepspeed_zero3_offload.yaml \
    --num_processes $num_gpus \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    tasks/train/train.py \
    tasks/train/${train_config}.py \
    deepspeed True \
    output_dir $output_dir \
    train_corpus $train_corpus \
    save_steps $save_steps \
    ckpt_steps $ckpt_steps \
    num_frames $num_frames \
    gradient_accumulation_steps $gradient_accumulation_steps \
    batch_size $batch_size \
    model.use_lora False \
    model.freeze_lm False \
    model.repo_id $repo_id \
    gradient_checkpointing True \
    optimizer.lr $lr \
    optimizer.weight_decay $wd \
    scheduler.epochs $epochs \
    scheduler.warmup_ratio $warmup_ratio \
    scheduler.min_lr_multi $min_lr \
    report_to wandb
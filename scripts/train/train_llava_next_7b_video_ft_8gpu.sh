#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Experiment
expr_name=llava_next_7b_video_ft_8-12-12
train_config=config_llava_next
train_corpus=videochat2_video
output_dir=workspace/outputs/${expr_name}

# Model
repo_id=workspace/models/llava-v1.6-vicuna-7b-hf
num_frames=8
pooling_shape="(8,12,12)"

# Training scale
num_gpus=8
full_batch_size=128 # Keep full_batch_size == num_gpus * batch_size * gradient_accumulation_steps
batch_size=16
gradient_accumulation_steps=1
num_save_samples=100000

# Hyperparameters
epochs=3
lr=2e-5
warmup_ratio=0.2
min_lr=0.25

# Python path
which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.

# Compute steps
save_steps=$(($num_save_samples/($batch_size*$num_gpus)))
ckpt_steps=$(($save_steps/10))

# Output dir
OUTPUT_DIR=workspace/outputs/${expr_name}

# Debug info
echo "PYTHONPATH: ${PYTHONPATH}"
echo "which python: ${which_python}"
echo "full_batch_size: ${full_batch_size}"
echo "batch_size: ${batch_size}"
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"
echo "save_steps: ${save_steps}"
echo "ckpt_steps: ${ckpt_steps}"
echo "train corpus: ${train_corpus}"

# Launch training
accelerate launch \
    --config_file scripts/train/accel_config_deepspeed_zero3_offload.yaml \
    --num_processes $num_gpus \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    tasks/train/train.py \
    tasks/train/${train_config}.py \
    output_dir $output_dir \
    train_corpus $train_corpus \
    save_steps $save_steps \
    ckpt_steps $ckpt_steps \
    num_workers 8 \
    num_frames $num_frames \
    deepspeed True \
    gradient_accumulation_steps $gradient_accumulation_steps \
    batch_size $batch_size \
    model.pooling_method avg \
    model.use_lora False \
    model.freeze_lm False \
    model.use_pooling True \
    model.repo_id $repo_id \
    gradient_checkpointing True \
    preprocess.center_pad False \
    preprocess.clip_transform False \
    optimizer.lr $lr \
    scheduler.epochs $epochs \
    scheduler.warmup_ratio $warmup_ratio \
    scheduler.min_lr_multi $min_lr \
    model.pooling_shape $pooling_shape \
    scheduler.is_videochat2_custom True \
    preprocess.mm_alone False \
    preprocess.random_shuffle False \
    preprocess.add_second_msg False \
    report_to wandb
#!/bin/bash
# Attention map visualization (Baseline vs. No cross-frame interactions) for LLaVA-NexT-7B-Video-FT in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: Please modify the paths and task ID before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/attention_visualization
video_model_path=workspace/models/LLaVA-NeXT-7B-Video-FT
conv_mode=eval_mvbench
pooling_shape=8-12-12
task_id=0

# Baseline vs. No cross-frame interactions
python analysis/attention_visualization.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape}
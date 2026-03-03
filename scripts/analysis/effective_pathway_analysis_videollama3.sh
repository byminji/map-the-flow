#!/bin/bash
# Effective pathway analysis for VideoLLaMA3 in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: Please modify the paths and task ID before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/effective_pathway_analysis
video_model_path=workspace/models/VideoLLaMA3-7B
data_root=/your/path/to/TVBench
max_frames=8
max_visual_tokens=$(( max_frames * 12 * 12 ))
task_id=0

# Blocking except for the effective pathway
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/effective_pathway_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target effective-pathway


# Random blocking
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/effective_pathway_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target random --random_block_ratio 0.42
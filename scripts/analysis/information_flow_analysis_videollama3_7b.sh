#!/bin/bash
# Information flow analysis for VideoLLaMA3 in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: Please modify the paths and task ID before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/information_flow_analysis
video_model_path=workspace/models/VideoLLaMA3-7B
data_root=/your/path/to/datasets/TVBench
max_frames=8
max_visual_tokens=$(( max_frames * 12 * 12 ))
task_id=0

# Cross-frame interactions (VideoLLM vs. ImageLLM)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/information_flow_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target cross-frame

# Cross-modal information flow (Video -> Question, Video -> Last, Question -> Last, Last -> Last)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/information_flow_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target vql-to-ql

# Question to Last flow (Non-option question, True option, False option -> Last)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/information_flow_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target question-and-options-to-last

# Video to True option flow (Video -> Non-option question, Non-option question -> True option, Video -> True option)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/information_flow_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}  \
--target vq-to-true-opt

# Generation probability tracing
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 analysis/gen_prob_analysis_videollama3.py \
--output_dir ${output_path} --model_path=${video_model_path} \
--benchmark ${dataset_name} --data_root ${data_root} --max_frames ${max_frames} \
--max_visual_tokens=${max_visual_tokens} --num_workers=1 --task_id ${task_id}
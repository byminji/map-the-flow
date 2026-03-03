#!/bin/bash
# Information flow analysis for LLaVA-NexT-13B-Video-FT in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: Please modify the paths and task ID before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/information_flow_analysis
video_model_path=workspace/models/Mini-InternVL-4B-Video-FT
base_model_path=workspace/models/Mini-InternVL-Chat-4B-V1-5
conv_mode=eval_mvbench_phi3
pooling_shape=8-16-16
task_id=0

# Cross-frame interactions (VideoLLM vs. ImageLLM)
python analysis/information_flow_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target cross-frame

python analysis/information_flow_analysis.py --output_dir ${output_path} --model_path ${base_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target cross-frame

# Cross-modal information flow (Video -> Question, Video -> Last, Question -> Last, Last -> Last)
python analysis/information_flow_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target vql-to-ql

# Question to Last flow (Non-option question, True option, False option -> Last)
python analysis/information_flow_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target question-and-options-to-last

# Video to True option flow (Video -> Non-option question, Non-option question -> True option, Video -> True option)
python analysis/information_flow_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target vq-to-true-opt

# Generation probability tracing
python gen_prob_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape}
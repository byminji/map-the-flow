#!/bin/bash
# Effective pathway analysis for LLaVA-NexT-13B-Video-FT in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: Please modify the paths and task ID before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/effective_pathway_analysis
video_model_path=workspace/models/Mini-InternVL-4B-Video-FT
conv_mode=eval_mvbench_phi3
pooling_shape=8-16-16
task_id=0

# Blocking except for the effective pathway
python analysis/effective_pathway_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target effective-pathway-internvl

# Random blocking
python analysis/effective_pathway_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--target random --random_block_ratio 0.6

#!/bin/bash
# Information flow analysis for LLaVA-NexT-7B-Video-FT in VCGBench in open-ended QA formats
# Task IDs: 2=temporal qa
# Note: Please modify the paths before running the script.
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=vcgbench
output_path=workspace/outputs/information_flow_analysis_open_ended
video_model_path=workspace/models/LLaVA-NeXT-7B-Video-FT
conv_mode=eval_vcgbench
pooling_shape=8-12-12
task_id=2

# First anchor generation
python analysis/information_flow_analysis_open_ended.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--generation_anchor_order 1

# Second anchor generation
python analysis/information_flow_analysis_open_ended.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--generation_anchor_order 2

# Third anchor generation
python analysis/information_flow_analysis_open_ended.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--generation_anchor_order 3

# To merge these plots in one graph, use visualize_multiple_plots_pretty

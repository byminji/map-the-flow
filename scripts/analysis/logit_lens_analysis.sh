#!/bin/bash
# Logit Lens analysis for LLaVA-NexT-13B-Video-FT in TVBench
# Task IDs: 0=Action Antonym, 3=Action Sequence, 5=Moving Direction, 6=Object Count, 8=Scene Transition, -1=Full dataset
# Note: We use 3=Action Sequence task in our paper
# Results will be saved under ${output_path}/${dataset_name}/${target}/${model_name}

dataset_name=tvbench
output_path=workspace/outputs/logit_lens_analysis
video_model_path=workspace/models/LLaVA-NeXT-13B-Video-FT
conv_mode=eval_mvbench
pooling_shape=8-12-12
task_id=3

# In case of saving logits over all samples
python analysis/logit_lens_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape}

# In case of visualization over video frames (e.g., Fig. 5 in our paper)
python analysis/logit_lens_analysis.py --output_dir ${output_path} --model_path ${video_model_path} \
--conv_mode ${conv_mode} --dataset_name ${dataset_name} --task_id ${task_id} --pooling_shape ${pooling_shape} \
--visualize_on_video

# Visualization of layerwise frequency plot after using precomputed logit lens results
python scripts/visualize_logit_lens_vocab_frequency.py --output_root ${output_path} --filename /path/to/saved/json
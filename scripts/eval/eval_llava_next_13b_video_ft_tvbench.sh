export CUDA_VISIBLE_DEVICES=0,1,2,3

model_dir=workspace/models/LLaVA-NeXT-13B-Video-FT
weight_dir=workspace/models/LLaVA-NeXT-13B-Video-FT

# In case of using your own trained model,
#model_dir=workspace/models/llava-v1.6-vicuna-13b-hf # base model path
#weight_dir=workspace/models/your_finetuned_model   # finetuned model path

SAVE_DIR=workspace/test_results/test_llava_next_13b_video_ft
lora_alpha=0
num_frames=8

conv_mode=eval_mvbench
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --is_tvbench \
    --conv_mode ${conv_mode} \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/tvbench \
    --lora_alpha ${lora_alpha} \
    --num_frames ${num_frames} \
    --weight_dir ${weight_dir} \
    --pooling_shape 8-12-12

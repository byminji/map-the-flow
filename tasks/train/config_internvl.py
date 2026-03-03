from tasks.train.instruction_data import *

# ========================= data ==========================
train_corpus = "videochat2_instruction_full"

train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict()
test_types = []
num_workers = 8
save_steps=10000
ckpt_steps=1000
stop_key = None
deepspeed=False
# ========================= input ==========================
num_frames = 12
num_frames_test = 1
batch_size = 1
gradient_accumulation_steps=16
max_txt_l = 512
max_train_steps=None
pre_text = False
inputs = dict(
    image_res=448,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model_type = 'internvl'
model = dict(
    repo_id="OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
    pretrained_path=None,
    load_from_origin=False,
    origin_vision="",
    origin_llm="",
    vision_encoder=dict(
        name="intern_vit_l14", # somehow need this to tell the dataset the mean std of pretrained model
        drop_path_rate=0.0,
    ),
    torch_dtype='bfloat16',
    freeze_projector=False,
    freeze_lm=False,
    freeze_vision_tower=True,
    use_lora=False,
    lora_r=0,
    lora_alpha=0,
    lora_dropout=0,
    num_frames="${num_frames}",
    down_sample_ratio=0.5, # (32, 32) -> (16, 16)
)
preprocess = dict(
    system="",
    mm_alone=False,
    random_shuffle=False,
    add_second_msg=False,
    roles=['<|user|>\n', '<|assistant|>\n'],
    end_signal=(' ', '<|end|>'),
    begin_signal='',
    dataset_image_placeholder='<Image></Image>',
    dataset_video_placeholder='<Video></Video>',
    image_token_index=32013,    # <IMG_CONTEXT>
    max_txt_l = "${max_txt_l}",
    ignore_index=-100, # same as torch softmax ignore index 
    center_pad=False,
    clip_transform=False,
    num_frames="${num_frames}",
)

optimizer = dict(
    opt="adamW",
    lr=4e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.05,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(
    is_videochat2_custom=True,
    sched="cosine", 
    epochs=1,
    warmup_ratio=0.03,
    min_lr_multi=0)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
project_name="InternVL"
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="${project_name}",  # setup in your command line
    dir="/tmp"   # directory to store logs locally
)
dist_url = "env://"
device = "cuda"
mode = "it"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 5
metric_window_size=10 # window size for metric
seed = 42
report_to='tensorboard'
save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?

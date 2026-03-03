
import torch
import os
from peft import get_peft_model, LoraConfig, TaskType
from safetensors import safe_open
from peft import PeftModel
from tasks.eval.eval_utils import Conversation

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model
from accelerate.utils import get_balanced_memory

from transformers import StoppingCriteria, AutoTokenizer, CLIPImageProcessor

from models.pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from models.internvl import InternVLChatConfig, InternVLChatModel

from tasks.shared_utils import INTERNVL_IMG_CONTEXT_TOKEN, IMAGENET_MEAN, IMAGENET_STD
from tasks.eval.mvbench import TVBenchDataset
from tasks.eval.tomato import TOMATODataset
from tasks.eval.longvideobench import LongVideoBenchDataset
from tasks.eval.videomme import VideoMMEDataset
from tasks.eval.vcgbench import VideoChatGPTBenchDataset


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha,
                           weight_dir, pooling_shape=(8, 12, 12), dataset_name='tvbench', dataset_path=None,
                           lora_target_modules=('q_proj', 'v_proj'), force_eager=True):
    if 'llava' in pretrained_model_name_or_path.lower():
        model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora,
                                       weight_dir=weight_dir, lora_alpha=lora_alpha, pooling_shape=pooling_shape,
                                       lora_target_modules=lora_target_modules,
                                       force_eager=force_eager)
    elif 'internvl' in pretrained_model_name_or_path.lower():
        model, processor = load_internvl(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora,
                                         weight_dir=weight_dir, lora_alpha=lora_alpha,
                                         force_eager=force_eager)
    else:
        raise NotImplementedError

    model = model.to(torch.device(rank))
    model = model.eval()

    if dataset_name == 'tvbench':
        dataset = TVBenchDataset(num_segments=num_frames)
    elif dataset_name == 'tvbench_open_ended':
        dataset = TVBenchDataset(num_segments=num_frames, open_ended=True)
    elif dataset_name == 'tomato':
        dataset = TOMATODataset(num_segments=num_frames)
    elif dataset_name == "longvideobench":
        dataset = LongVideoBenchDataset(insert_text=False, max_num_frames=num_frames)
    elif dataset_name == 'videomme':
        dataset = VideoMMEDataset(num_segments=num_frames)
    elif dataset_name == 'videomme_open_ended':
        dataset = VideoMMEDataset(num_segments=num_frames, open_ended=True)
    elif dataset_name == 'vcgbench':
        dataset = VideoChatGPTBenchDataset(num_segments=num_frames)
    else:
        raise NotImplementedError

    if dataset_name != "longvideobench":
        dataset.set_rank_and_world_size(rank, world_size)

    return model, processor, dataset


def load_pllava(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False,
                pooling_shape=(16,12,12), architectures=None,
                force_eager=False, lora_target_modules=("q_proj", "v_proj")):
    kwargs = {
        'num_frames': num_frames,
        'force_eager': force_eager
    }
    config = PllavaConfig.from_pretrained(
        repo_id if not use_lora else weight_dir,
        pooling_shape=pooling_shape,
        **kwargs,
    )

    # Explicitly update the architecture field in the loaded config
    if architectures is not None:
        config.text_config.architectures = [architectures]
    config.torch_dtype = torch.bfloat16  # Force torch.bfloat16

    with torch.no_grad():
        model = PllavaForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        
    try:
        processor = PllavaProcessor.from_pretrained(repo_id)
    except Exception as e:
        processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

    # config lora
    if use_lora and weight_dir is not None:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,  target_modules=lora_target_modules,
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha/128)
        model.language_model = get_peft_model(model.language_model, peft_config)
        assert weight_dir is not None, "pass a folder to your lora weight"
        print("Finish use lora")
    
    # load weights
    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full=True        
                    break
        else:
            use_full= True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
            
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
        print(model.hf_device_map)

    model = model.eval()

    return model, processor


def load_internvl(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False,
                  force_eager=False):
    try:
        # Use processor saved in fine-tuned model path
        print(f"Loading processor from {weight_dir}")
        processor = PllavaProcessor.from_pretrained(weight_dir)
    except Exception as e:
        print(f"Building processor by loading tokenizer from {repo_id} and building CLIPImageProcessor")
        # Individually load tokenizer and build image processor
        tokenizer = AutoTokenizer.from_pretrained(repo_id, add_eos_token=False, trust_remote_code=True, use_fast=False)

        # 1.2 Build image processor
        image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge": 448},
                                             do_center_crop=True, crop_size={"height": 448, "width": 448},
                                             do_rescale=True,  # default rescale factor is 1/255
                                             do_normalize=True, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD)

        processor = PllavaProcessor(tokenizer=tokenizer, image_processor=image_processor,
                                    shortest_edge=448, longest_edge=896)

    # Load model config
    img_context_token_id = processor.tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
    config = InternVLChatConfig.from_pretrained(repo_id if not use_lora else weight_dir,
                                                img_context_token_id=img_context_token_id,
                                                ignore_index=-100,
                                                num_frames=num_frames,
                                                force_eager=force_eager,
                                                )

    # Load model
    with torch.no_grad():
        model = InternVLChatModel.from_pretrained(repo_id, torch_dtype=torch.bfloat16, config=config)

    # Manually force InternLM's generation config into InternVLChat
    model.generation_config.pad_token_id = processor.tokenizer.unk_token_id    # 0: <unk>

    # config lora
    if use_lora and weight_dir is not None:
        model.wrap_llm_lora(r=16, lora_alpha=lora_alpha, lora_dropout=0.)
        model.config.use_llm_lora = use_lora
        assert weight_dir is not None, "pass a folder to your lora weight"

    # load weights
    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full = True
                    break
        else:
            use_full = True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)

        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                                     'Phi3DecoderLayer', 'Qwen2DecoderLayer'],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                                     'Phi3DecoderLayer', 'Qwen2DecoderLayer'],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
        print(model.hf_device_map)

    model = model.eval()

    return model, processor


def load_adapters(model, adapter_model_name_or_paths):

    for adapter_model_name_or_path in adapter_model_name_or_paths:
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, adapter_model_name_or_path, adapter_model_name_or_path)
        else:
            model.load_adapter(adapter_model_name_or_path, adapter_model_name_or_path)

    return model


def pllava_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, images=img_list, return_tensors="pt")
    # assert inputs['pixel_values'] is not None
    # if inputs['pixel_values'] is None:
    #     inputs.pop('pixel_values')
    inputs = inputs.to(model.device)

    """
    prompt
        <system prompt> + <user query> + <assistant response>
        = <system prompt> + "USER:" + <image token> + "USER:" + <question> + "ASSISTANT:" + <response_template>

        e.g.,
        Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects,
        and the action and pose of persons. Based on your observations, select the best option that accurately
        addresses the question.
         USER: <image>
         USER: Question: What happened after the person took the food?
        Options:
        (A) Ate the medicine.
        (B) Tidied up the blanket.
        (C) Put down the cup/glass/bottle.
        (D) Took the box.
        Only give the best option. ASSISTANT:Best option:(

    inputs
        - pixel_values: Tensor(T, 3, H, W)  e.g., Tensor(16, 3, 336, 336)
        - input_ids: Tensor(1, seq_len)
        - attention_mask: Tensor(1, seq_len)
    """

    # set up stopping criteria
    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria= None

    with torch.no_grad():
        # Next token prediction: https://github.com/huggingface/transformers/blob/745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec/src/transformers/generation/utils.py#L1284
        # generate() returns all input_ids generated till we meet the stopping criteria
        # e.g., input_ids = Tensor(1, 136) -> output_token = Tensor(1, 148): generated answer's token length is 12
        output_token = model.generate(**inputs, media_type='video',
                                      do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                      top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                      stopping_criteria=stopping_criteria)
        # Decode into text
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]

    # Gather answers without prompt inputs
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()
    conv.messages[-1][1] = output_text
    return output_text, conv


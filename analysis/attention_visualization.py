import argparse
from collections import defaultdict, Counter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from einops import rearrange

import torch
from tqdm import tqdm
import json
import random

torch.set_grad_enabled(False)
tqdm.pandas()

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(context="notebook",
        rc={"font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset

from causal_intervention_tools import (precompute_attention_masks, generate_with_attn_block, decode_tokens,
                                       generate_from_input, find_token_range, find_inter_frame_block_ranges)


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Attention visualization (Baseline vs. No cross-frame interactions)"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model results. e.g., workspace/visualization/information_flow_analysis")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory. e.g., workspace/models/LLaVA-NeXT-7B-Video-FT")
    parser.add_argument("--pooling_shape", type=str, required=True,
                        help="Pooling shape as a comma-separated string, e.g., '8-12-12'.")
    parser.add_argument("--conv_mode", type=str, required=True,
                        help="Conv mode. e.g., eval_mvbench")
    parser.add_argument("--dataset_name", type=str, default='tvbench',
                        help="Dataset name. e.g., tvbench")

    parser.add_argument('--weight_dir', type=str, default=None,
                        help="Path to the finetuned model weight.")
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_target_modules", type=parse_list, default=["q_proj", "v_proj"])

    parser.add_argument("--head_pooling", type=str, default='mean',
                        help="Attention head pooling")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Cross-frame attention blocking layer range. Default is all layers")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")

    parser.add_argument("--skip_baseline_wrong_samples", action='store_true')
    parser.add_argument("--skip_knockout_correct_samples", action='store_true')

    args = parser.parse_args()

    # Print args
    model_path = args.model_path
    pooling_shape = tuple(map(int, args.pooling_shape.split('-')))
    print(f'{model_path=}, {pooling_shape=}')
    print(f'{args.conv_mode=}, {args.target=}')

    weight_dir = args.model_path if args.weight_dir is None else args.weight_dir
    use_lora = (args.lora_alpha > 0)

    " ====== Load model ====== "
    dataset_name = args.dataset_name
    model, processor, dataset = load_model_and_dataset(rank=0,
                                                       world_size=1,
                                                       pretrained_model_name_or_path=model_path,
                                                       num_frames=pooling_shape[0],
                                                       use_lora=use_lora,
                                                       weight_dir=weight_dir,
                                                       dataset_name=dataset_name,
                                                       lora_alpha=args.lora_alpha,
                                                       lora_target_modules=args.lora_target_modules,
                                                       pooling_shape=pooling_shape,
                                                       force_eager=(not args.eval_only))

    if 'llava' in model_path.lower():
        num_layers = model.config.text_config.num_hidden_layers
    elif 'internvl' in model_path.lower():
        num_layers = model.config.llm_config.num_hidden_layers
    else:
        raise NotImplementedError

    conv_mode = args.conv_mode
    open_ended = (dataset_name in ['tvbench_open_ended', 'videomme_open_ended', 'vcgbench'])
    pre_query_prompt = None
    if open_ended:
        post_query_prompt = None
        answer_prompt = None
    elif dataset_name == 'videomme':
        post_query_prompt = "\nOnly give the best option."
        answer_prompt = "Best option:"
    else:
        post_query_prompt = "\nOnly give the best option."
        answer_prompt = "Best option:("

    if args.weight_dir is None:
        model_name = os.path.basename(model_path)
    else:
        model_name = os.path.basename(args.weight_dir)

    output_root = f'{args.output_dir}/{dataset_name}/{model_name}'
    print(f'{output_root=}')
    os.makedirs(output_root, exist_ok=True)

    vocab_counter = Counter()

    " ====== Arrange data ====== "
    # Arrange data by task type
    video_index_map = defaultdict(list)
    for idx, entry in enumerate(dataset.data_list):
        task_type = entry['task_type']
        video_index_map[task_type].append(idx)

    # Convert defaultdict to a sorted dictionary
    video_index_map = dict(sorted(video_index_map.items()))

    " ====== Start visualization ====== "
    for task_i, (task_type, qa_indices) in enumerate(video_index_map.items()):  # Iterate by task type
        if args.task_id != -1 and task_i != args.task_id:
            continue
        if args.test_ratio > 0:
            qa_indices = random.sample(qa_indices, min(args.test_ratio, len(qa_indices)))

        "============= Attention visualization analysis ============="
        # Run attention knockouts
        acc_base, cnt_samples = 0, 0
        with (tqdm(total=len(qa_indices), desc=f"Processing QA Samples for {task_type}", unit="sample") as pbar):
            for i, data_idx in enumerate(qa_indices):
                example = dataset[data_idx]

                "============= Prepare inputs ============="
                # Prepare prompt
                video_list = example["video_pils"]  # list(frame_length) of PIL Image
                conv = conv_templates[conv_mode].copy()
                conv.user_query(example['question'], pre_query_prompt, post_query_prompt, is_mm=True)

                if args.force_no_answer_prompt:
                    answer_prompt = None
                elif dataset_name == 'tvbench_open_ended':
                    answer_prompt = example['answer_prompt']

                if answer_prompt is not None:
                    conv.assistant_response(answer_prompt)
                stop_criteria_keywords = ["###", "USER"]
                if conv.roles[-1] == "<|im_start|>assistant\n":
                    split_tag = "<|im_start|> assistant\n"
                else:
                    split_tag = conv.roles[-1]

                # Prepare inputs
                torch.cuda.empty_cache()
                prompt = conv.get_prompt()
                inputs = processor(text=prompt, images=video_list, return_tensors="pt").to(model.device)
                inputs['media_type'] = 'video'  # Needed for PLLaVA

                "============= Define token ranges ============="
                """
                Define token ranges for system, vision, question, last

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

                    ->
                    system: Carefully ~ question.
                    vision: <image>
                    question: Question: What happend ~ (D) Took the box.
                """

                input_ids = inputs["input_ids"][0]

                # system
                # system_range = list(range(*find_token_range(processor.tokenizer, input_ids, conv.system)))

                # vision -> (<image> idx, <image> idx + num_vision_tokens)
                # because #num_vision_tokens tokens are inserted in the position of <image> token
                image_placeholder_index = torch.where(input_ids == model.config.image_token_index)[0].item()
                num_vis = pooling_shape[0] * pooling_shape[1] * pooling_shape[2]
                vision_range = [x + image_placeholder_index for x in range(num_vis)]

                # total prompt after vision tokens, except for the last input token
                ntoks = inputs["input_ids"].shape[1] + num_vis - 1
                last_token = ntoks - 1  # last token position

                # question -> should shift by (num_vision_tokens - 1)
                question_range = find_token_range(processor.tokenizer, input_ids, example['question'])
                assert question_range[0] > -1
                # assert reverse_check_token_range(processor.tokenizer, input_ids, question_range[0], question_range[1],
                #                                  example['question'])
                question_range = [x + (num_vis - 1) for x in range(question_range[0], question_range[1])]

                post_prompt_range = [j for j in range(question_range[0], last_token)]

                if answer_prompt is not None:
                    answer_prompt_range = find_token_range(processor.tokenizer, input_ids, answer_prompt)
                    assert answer_prompt_range[0] > -1
                    answer_prompt_range = [x + (num_vis - 1) for x in
                                           range(answer_prompt_range[0], answer_prompt_range[1])]
                    if answer_prompt_range[-1] == last_token:
                        answer_prompt_range = answer_prompt_range[:-1]

                if 'question_without_options' in example:
                    # question without options
                    q_rng = find_token_range(processor.tokenizer, input_ids, example['question_without_options'])
                    assert q_rng[0] > -1
                    question_without_options_range = [x + (num_vis - 1) for x in range(q_rng[0], q_rng[1])]

                    if open_ended:
                        tr_op = None
                    else:
                        # true option
                        tr_op = find_token_range(processor.tokenizer, input_ids, example['answer'])
                        assert tr_op[0] > -1
                        true_option_range = [x + (num_vis - 1) for x in range(tr_op[0], tr_op[1])]

                        # false options
                        false_options_range = []
                        fl_ops = []
                        for option in example['false_options']:
                            fl_op = find_token_range(processor.tokenizer, input_ids, option)
                            assert fl_op[0] > -1
                            false_options_range.extend([x + (num_vis - 1) for x in range(fl_op[0], fl_op[1])])
                            fl_ops.append(fl_op)

                "============= Baseline forward without blocking ============="
                # prediction
                answer_t, base_score, probs, output_text, attentions = generate_from_input(model, processor, inputs, conv, split_tag)

                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                # get correct token probability
                if open_ended:
                    gt_ids = processor.tokenizer(example["answer"], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                    gts = decode_tokens(processor.tokenizer, gt_ids)
                    for gt, gt_t in zip(gts, gt_ids):
                        if gt != "":  # non-empty first token
                            break

                else:
                    gt = example["answer"][1]  # e.g., 'A'
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                base_score_gt = probs[gt_t].cpu().item()

                acc_base += 1 if answer.lower() == gt.lower() else 0
                cnt_samples += 1

                if args.skip_baseline_wrong_samples and answer.lower() != gt.lower():
                    print("Skipping baseline's wrong sample")
                    continue

                "============= Forward with cross-frame attention blocking ============="
                query_lists, key_lists = find_inter_frame_block_ranges(vision_range,
                                                                       num_frames=pooling_shape[0],
                                                                       num_vis_one_frame=pooling_shape[1] *
                                                                                         pooling_shape[2],
                                                                       vis_start_id=image_placeholder_index)

                attn_mask = precompute_attention_masks(ntoks, model.config.text_config.num_attention_heads,
                                                       query_lists, key_lists, model.dtype, model.device)

                if args.layer_range is not None:
                    layer_range = tuple(map(int, args.layer_range.split('-')))
                    layerlist = [l for l in range(layer_range[0], layer_range[1])]
                else:
                    layerlist = [l for l in range(num_layers)]

                r_answer, r_gt, _, new_answer_t, new_output_text, new_attentions = generate_with_attn_block(
                    model, processor, inputs, conv, split_tag, answer_t, gt_t,
                    attn_mask, layerlist, use_lora)

                new_score = r_answer.cpu().item()
                new_score_gt = r_gt.cpu().item()
                new_answer_t = new_answer_t.cpu().item()
                [new_answer] = decode_tokens(processor.tokenizer, [new_answer_t])

                if args.skip_knockout_correct_samples and new_answer.lower() == gt.lower():
                    print("Skipping knockout correct sample")
                    continue

                "============= Headwise pooling ============="
                num_query = len(post_prompt_range)
                query_range = torch.tensor(post_prompt_range, device=model.device)
                key_range = torch.tensor(vision_range, device=model.device)
                q_indices = query_range[:, None]  # Make it a column vector
                s_indices = key_range[None, :]  # Make it a row vector

                base_attentions_pooled, new_attentions_pooled = [], []
                for layer in range(num_layers):
                    base_attn = attentions[layer][0, :, q_indices, s_indices]   # (h, len_q, len_k)
                    new_attn = new_attentions[layer][0, :, q_indices, s_indices]   # (h, len_q, len_k)

                    if args.head_pooling == 'mean':
                        # headwise mean pooling
                        base_attn = base_attn.mean(axis=0)
                        new_attn = new_attn.mean(axis=0)
                    elif args.head_pooling == 'max':
                        # headwise max pooling
                        base_attn = base_attn.max(axis=0)[0]    # (len_q, len_k)
                        new_attn = new_attn.max(axis=0)[0]      # (len_q, len_k)
                    else:
                        raise NotImplementedError

                    base_attentions_pooled.append(base_attn)
                    new_attentions_pooled.append(new_attn)
                base_attentions_pooled = torch.stack(base_attentions_pooled, dim=0) # (num_layer, len_q, len_k)
                new_attentions_pooled = torch.stack(new_attentions_pooled, dim=0) # (num_layer, len_q, len_k)

                "============= Save frame images as (1, num_frames * width) ============="
                def save_frame_images(fig_name):
                    # rows, cols = pooling_shape[0], 1  # Fixed 8x1 grid for 8 frames
                    frames_without_norm = processor.preprocess_masks(masks=video_list,
                                                                     return_tensors="pt")['mask_values'].to(model.device)

                    # grid with line
                    t, c, h, w = frames_without_norm.size()
                    frames_without_norm = rearrange(frames_without_norm, 't c h w -> h (t w) c')
                    frames_without_norm = frames_without_norm.cpu().numpy()
                    # plt.figure(figsize=(2, 16))
                    plt.figure(figsize=(16, 2))
                    plt.imshow(frames_without_norm)

                    num_frames = t
                    frame_borders = [k * h for k in range(num_frames + 1)]  # num_frames * h

                    for idx in frame_borders:
                        plt.axvline(x=idx - 0.5, color='white', linestyle='-', linewidth=4)  # Vertical line

                    # Remove x-axis and y-axis numbers
                    plt.xticks([])  # Remove x-axis numbers
                    plt.yticks([])  # Remove y-axis numbers

                    # Remove grid and borderlines
                    plt.grid(False)
                    plt.gca().spines['top'].set_visible(False)  # Remove top borderline
                    plt.gca().spines['right'].set_visible(False)  # Remove right borderline
                    plt.gca().spines['bottom'].set_visible(False)  # Remove bottom borderline
                    plt.gca().spines['left'].set_visible(False)  # Remove left borderline

                    # Make layout tight without borders
                    plt.tight_layout()

                    if fig_name is not None:
                        plt.savefig(fig_name)
                        print(f"Saved {fig_name}")

                    plt.show()
                    plt.close()

                save_root = f"{output_root}/{task_i:02d}_{task_type}/{i:03d}_{data_idx:05d}"
                os.makedirs(save_root, exist_ok=True)

                # save frame images
                save_frame_images(fig_name=f"{save_root}/frames.png")

                "============= Save attention map as (num_layers * height, num_frames * width) ============="
                def normalize_attention(attn, constant=5e-5):
                    # layerwise normalization
                    # Calculate min and max for each row (layer-wise)
                    attn = attn.float()

                    attn = rearrange(attn, 'l (n h w) -> (l h) (n w)',
                                     n=pooling_shape[0], h=pooling_shape[1], w=pooling_shape[2])

                    # Convert the tensor to numpy for plotting
                    attn = attn.detach().cpu().numpy()

                    # Apply log scale transformation
                    attn = np.log10(constant + attn)
                    return attn

                def visualize_attention(attn, title, min_val, max_val, cmap='viridis', layer_labels=None, fig_name=None):

                    # Plot figures
                    num_frames = pooling_shape[0]
                    h, w = pooling_shape[1], pooling_shape[2]
                    plt.figure(figsize=(16, 20))  # Adjust figure size as needed

                    # Plot the attention heatmap with log-scaled values
                    # plt.imshow(attn, cmap=cmap, aspect='auto', vmin=min_val, vmax=max_val)
                    plt.imshow(attn, cmap=cmap, vmin=min_val, vmax=max_val)

                    frame_borders = [k * w for k in range(num_frames + 1)]  # num_frames * w
                    layer_borders = [k * h for k in range(len(layer_labels) + 1)]  # num_layers * h

                    # x axis: layers (draw vertical lines between layers)
                    for idx in frame_borders:
                        plt.axvline(x=idx - 0.5, color='white', linestyle='-', linewidth=4)  # Horizontal line

                    # y axis: vision tokens (draw horizontal lines between tokens)
                    for idx in layer_borders:
                        plt.axhline(y=idx - 0.5, color='white', linestyle='-', linewidth=4)  # Vertical line

                    # Add axis labels
                    plt.ylabel('Layers', fontsize=30)
                    plt.xlabel('Frames', fontsize=30)

                    # Add tick labels
                    frame_labels = [str(k) for k in range(1, num_frames + 1)]
                    frame_ticks = [k * w + w // 2 for k in range(num_frames)]
                    plt.xticks(ticks=frame_ticks, labels=frame_labels, size=20)

                    layer_ticks = [k * h + h // 2 for k in range(len(layer_labels))]
                    plt.yticks(ticks=layer_ticks, labels=layer_labels, size=20)

                    # Add a title
                    plt.title(title, fontsize=20)

                    # Remove grid and borderlines
                    plt.grid(False)
                    plt.gca().spines['top'].set_visible(False)  # Remove top borderline
                    plt.gca().spines['right'].set_visible(False)  # Remove right borderline
                    plt.gca().spines['bottom'].set_visible(False)  # Remove bottom borderline
                    plt.gca().spines['left'].set_visible(False)  # Remove left borderline

                    # Make layout tight without borders
                    plt.tight_layout()

                    if fig_name is not None:
                        plt.savefig(fig_name)
                        print(f"Saved {fig_name}")

                    # Display the plot
                    plt.show()
                    plt.close()

                question_ids = input_ids[q_rng[0]:last_token]
                question_chars = decode_tokens(processor.tokenizer, question_ids)
                for op_id, op in enumerate(question_chars):
                    if op in " /()ABCD.?":
                        continue
                    constant = 5e-5

                    # Selective visualization in informative queries
                    if task_i == 0:
                        valid_vocab_list = ['Take', 'off', 'Put', 'on', 'W', 'ear', 'Stand', 'ing', 'up',
                                            'from', 'sitting', 'S', 'itting', 'down']
                        if op not in valid_vocab_list:
                            continue
                    elif task_i == 3:
                        valid_vocab_list = ['first', 'T', 'id', 'ied', 'up', 'Open', 'ed',
                             'Cl', 'osed', 'To', 'ok', 'D', 'rank', 'from',
                             'Put', 'down', 'Lied', 'on', 'Th', 'rew', 'Sat', 'H', 'eld']
                        if op not in valid_vocab_list:
                            continue
                    elif task_i == 5:
                        valid_vocab_list = ['direction', 'Down', 'right', 'left', 'Up']
                        if op not in valid_vocab_list:
                            continue
                    elif task_i == 8:
                        valid_vocab_list = ['From', 'to']
                        if op not in valid_vocab_list:
                            continue
                    elif task_i == 6:
                        if op.lower() not in ["begins", "ends"]:
                            continue
                    else:
                        raise NotImplementedError

                    base_attn = base_attentions_pooled[10:20, op_id, :] # Selective visualization in layers 10-20
                    new_attn = new_attentions_pooled[10:20, op_id, :]   # Selective visualization in layers 10-20
                    base_attn = normalize_attention(base_attn, constant)
                    new_attn = normalize_attention(new_attn, constant)

                    layer_labels = [str(k) for k in range(11, 21)]

                    visualize_attention(base_attn, f'Base (Query: "{op}")',
                                        # min_val, max_val,
                                        base_attn.min(), base_attn.max(),
                                        layer_labels=layer_labels,
                                        fig_name=f"{save_root}/query_{op_id:02d}_{op}_base.png")
                    visualize_attention(new_attn, f'No Cross-Frame Interactions (Query: "{op}")',
                                        # min_val, max_val,
                                        new_attn.min(), new_attn.max(),
                                        layer_labels=layer_labels,
                                        fig_name=f"{save_root}/query_{op_id:02d}_{op}_new.png")

                # save result
                result = {
                    "prompt": prompt,
                    "base_score": base_score,
                    "new_score": new_score,
                    "relative_diff": (new_score - base_score) * 100.0 / base_score,
                    # answer token probability drop
                    "gt_score": base_score_gt,
                    "new_score_gt": new_score_gt,
                    "relative_diff_gt": (new_score_gt - base_score_gt) * 100.0 / base_score_gt,
                    # gt token probability drop
                    "video_path": example['video_path'],
                    "data_id": data_idx,
                    "gt": gt,
                    "base_answer": answer,
                    "new_answer": new_answer
                }
                if open_ended:
                    result["gt_text"] = example["answer"]
                    result["base_output_text"] = output_text
                    result["new_output_text"] = new_output_text

                # Show last result in tqdm without breaking the progress bar
                tqdm.write(json.dumps(result, indent=4))

                # Save results as a file
                with open(f"{save_root}/result.json", 'w') as f:
                    json.dump(result, f, indent=4)

                pbar.update(1)  # Update progress after each QA sample


if __name__ == "__main__":
    main()

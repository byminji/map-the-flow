import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tqdm import tqdm
import json
import random

torch.set_grad_enabled(False)
tqdm.pandas()

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset

from analysis.causal_intervention_tools import (precompute_attention_masks, precompute_random_block_attention_masks,
                                                trace_with_attn_block, generate_with_attn_block,
                                                decode_tokens, predict_from_input, generate_from_input, find_token_range,
                                                find_inter_frame_block_ranges)


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Effective pathway analysis: "
                    "Apply attention edge pruning except for the target region. "
                    "We can suppress a substantial amount of edges (e.g., 58% in LLaVA-NeXT-7B-Video-FT) "
                    "with only marginal performance drop."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--pooling_shape", type=str, default='8-12-12',
                        help="Pooling shape as a dash-separated string, e.g., '8-12-12'.")
    parser.add_argument("--conv_mode", type=str, default='eval_mvbench',
                        help="Conv mode")
    parser.add_argument("--dataset_name", type=str, default='tvbench',
                        help="Dataset name")

    parser.add_argument('--weight_dir', type=str, default=None,
                        help="Path to the finetuned model weight.")
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_target_modules", type=parse_list, default=["q_proj", "v_proj"])

    parser.add_argument("--target", type=str, default='effective-pathway',
                        choices=["effective-pathway-7b", "effective-pathway-13b", "effective-pathway-internvl", "random"],
                        help="Target blocking mode")
    parser.add_argument("--random_block_ratio", type=float, default=0.0,
                        help="Random blocking ratio. Set from 0 to 1")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")
    parser.add_argument("--eval_only", action='store_true',
                        help="Eval mode without Attention Knockout")

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
        mask_num_heads = model.config.text_config.num_attention_heads
        image_token_index = model.config.image_token_index
        model_type = 'llava_lora' if use_lora else 'llava'
    elif 'internvl' in model_path.lower():
        num_layers = model.config.llm_config.num_hidden_layers
        mask_num_heads = 1
        image_token_index = model.img_context_token_id
        model_type = 'internvl_lora' if use_lora else 'internvl'
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

    output_root = f'{args.output_dir}/{dataset_name}/{args.target}/{model_name}'
    print(f'{output_root=}')
    os.makedirs(output_root, exist_ok=True)

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
            random.seed(42)
            qa_indices = random.sample(qa_indices, min(args.test_ratio, len(qa_indices)))

        "============= Information flow analysis ============="
        # Run attention knockouts
        acc_base, cnt_samples = 0, 0
        results = []
        with (tqdm(total=len(qa_indices), desc=f"Processing QA Samples for {task_type}", unit="sample") as pbar):
            for i, data_idx in enumerate(qa_indices):
                example = dataset[data_idx]

                "============= Prepare inputs ============="
                # Prepare prompt
                video_list = example["video_pils"]  # list(frame_length) of PIL Image
                conv = conv_templates[conv_mode].copy()
                conv.user_query(example['question'], pre_query_prompt, post_query_prompt, is_mm=True)

                if dataset_name == 'tvbench_open_ended':
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

                # vision -> (<image> idx, <image> idx + num_vision_tokens)
                # because #num_vision_tokens tokens are inserted in the position of <image> token
                image_placeholder_index = torch.where(input_ids == image_token_index)[0].item()
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

                "============= Baseline forward without blocking ============="
                # prediction
                if open_ended:
                    answer_t, base_score, probs, output_text = generate_from_input(model, processor, inputs,
                                                                                   conv, split_tag)
                else:
                    answer_t, base_score, probs = [d[0] for d in predict_from_input(model, inputs)]

                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                # get correct token probability
                if open_ended:
                    gt_ids = processor.tokenizer(example["answer"], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                    # Get first token only
                    gts = decode_tokens(processor.tokenizer, gt_ids)
                    for gt, gt_t in zip(gts, gt_ids):
                        if gt != "":    # non-empty first token
                            break

                else:
                    gt = example["answer"][1]  # e.g., 'A'
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                base_score_gt = probs[gt_t].cpu().item()

                acc_base += 1 if answer.lower() == gt.lower() else 0
                cnt_samples += 1

                torch.cuda.empty_cache()

                "============= Attention Knockout ============="
                query_lists, key_lists = find_inter_frame_block_ranges(vision_range,
                                                                       num_frames=pooling_shape[0],
                                                                       num_vis_one_frame=pooling_shape[1] *
                                                                                         pooling_shape[2],
                                                                       vis_start_id=image_placeholder_index)

                if 'effective-pathway' in args.target:
                    if args.target == 'effective-pathway-7b':
                        INTER_FRAME_RANGE = list(range(5, 15))          # L6-15
                        VIDEO_TO_QUESTION_RANGE = list(range(5, 20))    # L6-20
                        QUESTION_TO_LAST_RANGE = list(range(15, 25))    # L16-25
                    elif args.target == 'effective-pathway-13b':
                        INTER_FRAME_RANGE = list(range(5, 15))          # L6-15
                        VIDEO_TO_QUESTION_RANGE = list(range(5, 20))    # L6-20
                        QUESTION_TO_LAST_RANGE = list(range(15, 30))    # L16-30
                    elif args.target == 'effective-pathway-internvl':
                        INTER_FRAME_RANGE = list(range(5, 15))          # L6-15
                        VIDEO_TO_QUESTION_RANGE = list(range(5, 20))    # L6-20
                        QUESTION_TO_LAST_RANGE = list(range(10, 30))    # L11-30
                    else:
                        raise NotImplementedError

                    block_mappings = {
                        "vision_inter_frame": (key_lists, query_lists,
                                               [i for i in range(num_layers) if i not in INTER_FRAME_RANGE]),
                        "vision_to_question": ([vision_range], [question_range],
                                               [i for i in range(num_layers) if i not in VIDEO_TO_QUESTION_RANGE]),
                        "question_to_last": ([question_range], [[last_token]],
                                             [i for i in range(num_layers) if i not in QUESTION_TO_LAST_RANGE]),
                        "last_to_last": ([[last_token]], [[last_token]], [i for i in range(num_layers)]),
                        "vision_to_last": ([vision_range], [[last_token]], [i for i in range(num_layers)]),
                        "vision_to_vision": ([vision_range], [vision_range],
                                             [i for i in range(VIDEO_TO_QUESTION_RANGE[-1]+1, num_layers)]),
                        "question_to_question": ([question_range], [question_range],
                                                       [i for i in range(QUESTION_TO_LAST_RANGE[-1]+1, num_layers)])
                    }
                elif args.target == "random":
                    assert args.random_block_ratio > 0
                    num_baseline = ntoks * (ntoks + 1) // 2
                    num_block = int(num_baseline * args.random_block_ratio)
                    block_start = vision_range[0]
                    block_end = last_token + 1

                    def get_valid_indices(block_start, block_end, device='cpu'):
                        rows, cols = torch.tril_indices(row=block_end, col=block_end, offset=0, device=device)
                        mask = (rows >= block_start) & (cols >= block_start)
                        return torch.stack([rows[mask], cols[mask]], dim=1)  # shape [num_valid, 2]

                    valid_indices = get_valid_indices(block_start, block_end, device=model.device)
                else:
                    raise NotImplementedError

                # precompute attention mask
                attn_mask_all_layers = []
                attn_count_baseline, attn_count_new = 0, 0
                for layer in range(num_layers):
                    if args.target == "random":
                        attn_mask = precompute_random_block_attention_masks(ntoks,
                                                                            mask_num_heads,
                                                                            valid_indices=valid_indices,
                                                                            N=num_block, dtype=model.dtype,
                                                                            device=model.device)
                    else:
                        block_key_range, block_query_range = [], []
                        for block_desc, (keys, querys, layers) in block_mappings.items():
                            if layer not in layers:
                                continue
                            block_key_range.extend(keys)
                            block_query_range.extend(querys)

                        attn_mask = precompute_attention_masks(ntoks, mask_num_heads,
                                                               block_query_range, block_key_range, model.dtype,
                                                               model.device)

                    seq_len = attn_mask.size(-1)
                    attn_count_baseline += seq_len * (seq_len + 1) // 2
                    attn_count_new += (attn_mask[0, 0] == 0).sum().item()

                    attn_mask_all_layers.append(attn_mask)

                # attention knockout
                if open_ended:
                    r_answer, r_gt, _, new_answer_t, new_output_text = generate_with_attn_block(
                        model, processor, inputs, conv, split_tag, answer_t, gt_t,
                        attn_mask_all_layers, None, model_type)
                else:
                    r_answer, r_gt, _, new_answer_t = trace_with_attn_block(model, inputs, answer_t, gt_t,
                                                                            attn_mask_all_layers, None, model_type)

                new_score = r_answer.cpu().item()
                new_score_gt = r_gt.cpu().item()
                new_answer_t = new_answer_t.cpu().item()
                [new_answer] = decode_tokens(processor.tokenizer, [new_answer_t])

                results.append({
                    "prompt": prompt,
                    "block_desc": args.target,
                    "layer": 0,
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
                    "new_answer": new_answer,
                    # attention count
                    "base_attention_cnt": attn_count_baseline,
                    "new_attention_cnt": attn_count_new,
                    "attention_amount": attn_count_new / attn_count_baseline * 100
                })
                if open_ended:
                    results[-1]["gt_text"] = example["answer"]
                    results[-1]["base_output_text"] = output_text
                    results[-1]["new_output_text"] = new_output_text

                # Show last result in tqdm without breaking the progress bar
                tqdm.write(json.dumps(results[-1], indent=4))

                pbar.update(1)  # Update progress after each QA sample

                torch.cuda.empty_cache()

        "============= Visualization ============="

        # attention amount
        base_attention_cnt = sum(d['base_attention_cnt'] for d in results) / len(results)
        new_attention_cnt = sum(d['new_attention_cnt'] for d in results) / len(results)
        attention_amount = new_attention_cnt / base_attention_cnt * 100
        print(f"{base_attention_cnt=:.2f}, {new_attention_cnt=:.2f}, {attention_amount=:.2f}")

        acc_base = acc_base / cnt_samples * 100
        acc_new = defaultdict(lambda: defaultdict(float))
        for i, result in enumerate(results):
            block_desc = result['block_desc']
            layer = result['layer']
            if layer not in acc_new[block_desc]:
                acc_new[block_desc][layer] = 0
            if result['gt'] == result["new_answer"]:
                acc_new[block_desc][layer] += 1 / cnt_samples * 100
            results[i]['acc_base'] = acc_base
        for i, result in enumerate(results):
            results[i]['acc'] = acc_new[result['block_desc']][result['layer']]
            results[i]['base_attention_cnt_avg'] = base_attention_cnt
            results[i]['new_attention_cnt_avg'] = new_attention_cnt
            results[i]['attention_amount_avg'] = attention_amount

        # Save results as a file
        os.makedirs(f"{output_root}/jsons", exist_ok=True)
        with open(f"{output_root}/jsons/{task_i:02d}_{task_type}.json", 'w') as f:
            json.dump(results, f, indent=4)

        print(f"{acc_base=}, {acc_new=}")


if __name__ == "__main__":
    main()

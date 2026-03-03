import argparse
import os
import datetime
import numpy as np
import torch.distributed as dist
import itertools
import sys
from collections import defaultdict
import torch
from tqdm import tqdm
import json
import random

torch.set_grad_enabled(False)
tqdm.pandas()

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videollama3.videollama3 import disable_torch_init
from videollama3.evaluation.benchmarks import build_dataset
from videollama3.evaluation.register import INFERENCES
from analysis.causal_intervention_tools import (precompute_attention_masks, precompute_random_block_attention_masks,
                                                trace_with_attn_block, decode_tokens, predict_from_input, find_token_range)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Effective pathway analysis: "
                    "Apply attention edge pruning except for the target region. "
    )
    parser.add_argument("--model-path", "--model_path", type=str, default="workspace/models/VideoLLaMA3-7B")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--max-frames", "--max_frames", type=int, default=8)
    parser.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="workspace/effective_pathway_analysis")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")

    parser.add_argument("--target", type=str, default='effective-pathway',
                        choices=["effective-pathway", "random"],
                        help="Target blocking mode")
    parser.add_argument("--random_block_ratio", type=float, default=0.0,
                        help="Random blocking ratio. Set from 0 to 1")
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    seed_everything()
    args = parse_args()

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor = model_init(
        args.model_path,
        args.max_visual_tokens,
        device_map={"": f"cuda:{local_rank}"}
    )
    ########### DO NOT REMOVE
    model.config.use_token_compression = False
    processor.fps = None
    ########### DO NOT REMOVE

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        processor=processor,
        num_splits=dist.get_world_size(),
        split_idx=global_rank,
        fps=None,
        max_frames=args.max_frames,
    )

    dataset_name = args.benchmark
    model_name = args.model_path.split('/')[-1]
    output_root = f'{args.output_dir}/{dataset_name}/{args.target}/{model_name}'
    print(f'{output_root=}')
    os.makedirs(output_root, exist_ok=True)

    # configurations
    num_layers = 28
    mask_num_heads = 28

    " ====== Arrange data ====== "
    video_index_map = defaultdict(list)
    aggregated_data_list = dataset._aggregated_data_list
    for idx, entry in enumerate(aggregated_data_list):
        data_id = entry['data_ids'][0]
        task_type = dataset.data_dict[data_id]['task_type']
        video_index_map[task_type].append(idx)

    # Convert defaultdict to a sorted dictionary
    video_index_map = dict(sorted(video_index_map.items()))

    " ====== Start visualization ====== "
    for task_i, (task_type, qa_indices) in enumerate(video_index_map.items()):  # Iterate by task type
        if args.task_id != -1 and task_i != args.task_id:
            continue
        if args.test_ratio > 0:
            qa_indices = random.sample(qa_indices, min(args.test_ratio, len(qa_indices)))

        "============= Information flow analysis ============="
        # Run attention knockouts
        acc_base, cnt_samples = 0, 0
        results = []
        with (tqdm(total=len(qa_indices), desc=f"Processing QA Samples for {task_type}", unit="sample") as pbar):
            for video_idx in qa_indices:
                data = dataset[video_idx]
                data_ids = data["data_ids"]
                text_inputs = data["text_inputs"]
                for data_id, text_input in zip(data_ids, text_inputs):
                    data_dict = {**data["image_inputs"], **text_input}
                    data_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
                    if "pixel_values" in data_dict:
                        data_dict["pixel_values"] = data_dict["pixel_values"].to(torch.bfloat16)

                    "============= Baseline forward without blocking ============="
                    answer_t, base_score, probs = [d[0] for d in predict_from_input(model, data_dict)]

                    base_score = base_score.cpu().item()
                    [answer] = decode_tokens(processor.tokenizer, [answer_t])

                    gt = dataset.data_dict[data_id]['ground_truth']
                    if args.benchmark in ['tvbench', 'tomato']:
                        gt = chr(gt + 65)
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                    base_score_gt = probs[gt_t].cpu().item()

                    acc_base += 1 if answer.lower() == gt.lower() else 0
                    cnt_samples += 1

                    "============= Define token ranges ============="
                    # define token range
                    input_ids = data_dict['input_ids'][0]
                    prompt_without_image_tokens = processor.batch_decode(input_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
                    prompt = ''.join(prompt_without_image_tokens)

                    # vision token positions
                    # parse with "\n" = 198
                    borders = torch.where(input_ids == 198)[0]
                    vision_range = [i for i in range(borders[0] + 1, borders[1])]
                    last_token = len(input_ids) - 1

                    borders_frame = torch.where(input_ids == 11)[0]
                    range_sep_by_frames = []
                    for i in range(args.max_frames):
                        if i == 0:
                            range_sep_by_frames.append([x for x in range(borders[0] + 1, borders_frame[0])])
                        elif i == args.max_frames - 1:
                            range_sep_by_frames.append([x for x in range(borders_frame[i-1], borders[1])])
                        else:
                            range_sep_by_frames.append([x for x in range(borders_frame[i-1], borders_frame[i])])

                    query_lists, key_lists = [], []
                    for i in range(1, args.max_frames):
                        query_lists.append(range_sep_by_frames[i])
                        key_lists.append(
                            list(itertools.chain.from_iterable(range_sep_by_frames[:i])))  # block previous ranges

                    question_start_pos = borders[1] + 1
                    # question without options
                    question_str = dataset.data_dict[data_id]['question']+'\n'
                    q_rng = find_token_range(processor.tokenizer, input_ids[question_start_pos:], question_str,
                                             remove_margin=False)
                    assert q_rng[0] > -1

                    question_without_options_range = [x + question_start_pos for x in range(q_rng[0], q_rng[1])]

                    # true option
                    true_option_idx = dataset.data_dict[data_id]['ground_truth']
                    option_letters_full = dataset.data_dict[data_id]['option_letters_full']
                    true_option_str = option_letters_full[true_option_idx]
                    tr_op = find_token_range(processor.tokenizer, input_ids[question_start_pos:], true_option_str,
                                             remove_margin=False)
                    assert tr_op[0] > -1
                    true_option_range = [x + question_start_pos for x in range(tr_op[0], tr_op[1])]

                    # false options
                    false_options_range = []
                    fl_ops = []
                    for i, option in enumerate(option_letters_full):
                        if i == true_option_idx:
                            continue
                        fl_op = find_token_range(processor.tokenizer, input_ids[question_start_pos:], option,
                                                 remove_margin=False)
                        assert fl_op[0] > -1
                        false_options_range.extend([x + question_start_pos for x in range(fl_op[0], fl_op[1])])
                        fl_ops.append(fl_op)

                    question_end_pos = max(question_without_options_range+true_option_range+false_options_range)
                    question_range = [i for i in range(question_start_pos, question_end_pos+1)]
                    ntoks = len(input_ids)

                    "============= Attention Knockout ============="
                    if 'effective-pathway' in args.target:
                        INTER_FRAME_RANGE = list(range(0, 15))          # L1-15
                        VIDEO_TO_QUESTION_RANGE = list(range(5, 20))    # L6-20
                        QUESTION_TO_LAST_RANGE = list(range(20, 28))    # L21-28

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
                                                 [i for i in range(VIDEO_TO_QUESTION_RANGE[-1] + 1, num_layers)]),
                            "question_to_question": ([question_range], [question_range],
                                                     [i for i in range(QUESTION_TO_LAST_RANGE[-1] + 1, num_layers)])
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

                    r_answer, r_gt, _, new_answer_t = trace_with_attn_block(model, data_dict, answer_t, gt_t,
                                                                            attn_mask_all_layers, None,
                                                                            model_type='videollama')

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
                        "data_id": data_id,
                        "gt": gt,
                        "base_answer": answer,
                        "new_answer": new_answer,
                        # attention count
                        "base_attention_cnt": attn_count_baseline,
                        "new_attention_cnt": attn_count_new,
                        "attention_amount": attn_count_new / attn_count_baseline * 100
                    })

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

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

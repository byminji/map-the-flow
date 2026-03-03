import argparse
import os
import datetime
import numpy as np
import torch.distributed as dist
import itertools
import sys
from collections import defaultdict
import pandas as pd
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

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videollama3.videollama3 import disable_torch_init
from videollama3.evaluation.benchmarks import build_dataset
from videollama3.evaluation.register import INFERENCES
from analysis.causal_intervention_tools import (precompute_attention_masks, trace_with_attn_block,
                                                predict_from_input, decode_tokens, find_token_range)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Information flow analysis: "
                    "Trace the information flow dynamics by knocking out target attention regions."
    )
    parser.add_argument("--model-path", "--model_path", type=str, default="workspace/models/VideoLLaMA3-7B")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--max-frames", "--max_frames", type=int, default=8)
    parser.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="workspace/information_flow_analysis")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")

    parser.add_argument("--target", type=str, default='cross-frame',
                        choices=["cross-frame", "vql-to-ql", "question-and-options-to-last", "vq-to-true-opt"],
                        help="Target blocking mode")

    parser.add_argument("--sample_mode", type=str, default="correct",
                        choices=["correct", "wrong", "all"],
                        help="Sample gathering mode. Default is to analyze with only correctly answered samples.")

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
    window = 9
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
                    if args.benchmark == 'tvbench':
                        gt = chr(gt + 65)
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                    base_score_gt = probs[gt_t].cpu().item()

                    acc_base += 1 if answer.lower() == gt.lower() else 0
                    cnt_samples += 1

                    if args.sample_mode == "correct" and answer.lower() != gt.lower():
                        print("Skipping baseline wrong sample")
                        continue
                    if args.sample_mode == "wrong" and answer.lower() == gt.lower():
                        print("Skipping baseline correct sample")
                        continue

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
                    if args.benchmark == 'tvbench':
                        true_option_idx = dataset.data_dict[data_id]['ground_truth']
                    elif args.benchmark == 'longvideobench':
                        true_option_idx = ord(dataset.data_dict[data_id]['ground_truth']) - ord("A")
                    else:
                        raise NotImplementedError

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
                    if args.target == "cross-frame":
                        block_mappings = [(key_lists, query_lists, "No cross-frame interactions")]

                    elif args.target == "vql-to-ql":
                        block_mappings = [
                            ([vision_range], [question_range], "Video -/-> Question"),
                            ([vision_range], [[last_token]], "Video -/-> Last"),
                            ([question_range], [[last_token]], "Question -/-> Last"),
                            ([[last_token]], [[last_token]], "Last -/-> Last"),
                        ]

                    elif args.target == "question-and-options-to-last":
                        block_mappings = [
                            ([question_without_options_range], [[last_token]], "Non-option question -/-> Last"),
                            ([true_option_range], [[last_token]], "True option -/-> Last"),
                            ([false_options_range], [[last_token]], "False option -/-> Last")]

                    elif args.target == "vq-to-true-opt":
                        block_mappings = [
                            ([vision_range], [question_without_options_range], "Video -/-> Non-option question"),
                            ([question_without_options_range], [true_option_range],
                             "Non-option question -/-> True option"),
                            ([vision_range], [true_option_range], "Video -/-> True option")]

                    else:
                        raise NotImplementedError

                    for block_key_range, block_query_range, block_desc in block_mappings:
                        # # **Precompute the attention mask once for this block configuration**
                        attn_mask = precompute_attention_masks(ntoks, mask_num_heads,
                                                               block_query_range, block_key_range, model.dtype,
                                                               model.device)

                        for layer in range(num_layers):
                            layerlist = [l for l in
                                         range(max(0, layer - window // 2), min(num_layers, layer - (-window // 2)))]

                            # **Pass only to selected layers**
                            r_answer, r_gt, _, new_answer_t = trace_with_attn_block(model, data_dict, answer_t, gt_t,
                                                                                    attn_mask, layerlist,
                                                                                    model_type='videollama')

                            new_score = r_answer.cpu().item()
                            new_score_gt = r_gt.cpu().item()
                            new_answer_t = new_answer_t.cpu().item()
                            [new_answer] = decode_tokens(processor.tokenizer, [new_answer_t])

                            results.append({
                                "prompt": prompt,
                                "block_desc": block_desc,
                                "layer": layer,
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
                                "new_answer": new_answer
                            })

                            # Show last result in tqdm without breaking the progress bar
                            tqdm.write(json.dumps(results[-1], indent=4))

                            if len(layerlist) >= num_layers:
                                break

                    pbar.update(1)  # Update progress after each QA sample


        "============= Visualization ============="
        def plot_results(results_data, results_keyword, y_data_name, hline_val=0):
            tmp = pd.DataFrame.from_records(results_data)
            tmp["layer_1"] = tmp.layer.apply(lambda x: x + 1)

            plt.figure(figsize=(8, 6))
            ax = sns.lineplot(tmp, x="layer_1", y=y_data_name,
                              hue="block_desc",
                              style="block_desc",
                              dashes=True,
                              palette=palette, linewidth=1)
            ax.set_xlabel("layer")
            ax.set_ylabel(f"% change in {y_data_name}")
            ax.set_xlim(0, num_layers + 0.5)
            sns.move_legend(ax, "lower right", title="blocked positions")
            plt.axhline(y=hline_val, color=palette[2], linestyle='-')

            plt.savefig(f"{output_root}/{results_keyword}_{y_data_name}_target_{args.target}_"
                        f"{task_i:02d}_{task_type}.png")

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

        correct_results = [x for x in results if x["gt"].lower() == x["base_answer"].lower()]
        wrong_results = [x for x in results if x["gt"].lower() != x["base_answer"].lower()]
        plot_results(results, 'results', 'relative_diff')
        plot_results(results, 'results', 'acc', hline_val=acc_base)
        if args.sample_mode == "all" and len(correct_results) > 0:
            plot_results(correct_results, 'correct_results', 'relative_diff')
        if args.sample_mode == "all" and len(wrong_results) > 0:
            plot_results(wrong_results, 'wrong_results', 'relative_diff')

        # Save results as a file
        os.makedirs(f"{output_root}/jsons", exist_ok=True)
        with open(f"{output_root}/jsons/{task_i:02d}_{task_type}.json", 'w') as f:
            json.dump(results, f, indent=4)

        print(f"{acc_base=}, {acc_new=}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

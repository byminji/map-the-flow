import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import (precompute_attention_masks, trace_with_attn_block, generate_with_attn_block,
                                                decode_tokens, predict_from_input, generate_from_input, find_token_range,
                                                find_inter_frame_block_ranges)


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Information flow analysis: "
                    "Trace the information flow dynamics by knocking out target attention regions."
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

    parser.add_argument("--target", type=str, default='cross-frame',
                        choices=["cross-frame", "vql-to-ql", "question-and-options-to-last", "vq-to-true-opt"],
                        help="Target blocking mode")
    parser.add_argument("--window", type=int, default=9,
                        help="Blocking window size.")
    parser.add_argument("--window_style", type=str, default='center',
                        choices=["center", "inverse", "top_down", "bottom_up"],
                        help="Blocking window style.")
    parser.add_argument("--sweep_range", type=str, default=None,
                        help="Layer sweeping start and end")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")
    parser.add_argument("--sample_mode", type=str, default="correct",
                        choices=["correct", "wrong", "all"],
                        help="Sample gathering mode. Default is to analyze with only correctly answered samples.")
    parser.add_argument("--eval_only", action='store_true',
                        help="Eval mode without Attention Knockout")

    args = parser.parse_args()

    # Print args
    model_path = args.model_path
    pooling_shape = tuple(map(int, args.pooling_shape.split('-')))
    print(f'{model_path=}, {pooling_shape=}')
    print(f'{args.conv_mode=}, {args.target=}, {args.window=}')

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
    window = args.window
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

                "============= Baseline forward without blocking ============="
                # prediction
                if open_ended:
                    answer_t, base_score, probs, output_text = generate_from_input(model, processor, inputs,
                                                                                   conv, split_tag)
                    print(prompt)
                    print(output_text)
                    print(example['answer'])
                else:
                    answer_t, base_score, probs = [d[0] for d in predict_from_input(model, inputs)]

                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                # get correct token probability
                if open_ended:
                    gt_ids = processor.tokenizer(example["answer"], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                    gts = decode_tokens(processor.tokenizer, gt_ids)
                    for gt, gt_t in zip(gts, gt_ids):
                        if gt != "":    # non-empty first token
                            break

                else:
                    gt = example["answer"][1] if dataset_name != 'videomme' else example["answer"][0]  # e.g., 'A'
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                base_score_gt = probs[gt_t].cpu().item()

                acc_base += 1 if answer.lower() == gt.lower() else 0
                cnt_samples += 1

                if args.eval_only:
                    pbar.update(1)
                    continue
                else:
                    if args.sample_mode == "correct" and answer.lower() != gt.lower():
                        print("Skipping baseline wrong sample")
                        continue
                    if args.sample_mode == "wrong" and answer.lower() == gt.lower():
                        print("Skipping baseline correct sample")
                        continue

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

                "============= Attention Knockout ============="
                if args.target == "cross-frame":
                    query_lists, key_lists = find_inter_frame_block_ranges(vision_range,
                                                                           num_frames=pooling_shape[0],
                                                                           num_vis_one_frame=pooling_shape[1] *
                                                                                             pooling_shape[2],
                                                                           vis_start_id=image_placeholder_index)

                    block_mappings = [(key_lists, query_lists, "No cross-frame interactions")]

                elif args.target == "vql-to-ql":
                    block_mappings = [
                        ([vision_range], [question_range], "Video -/-> Question"),
                        ([vision_range], [[last_token]], "Video -/-> Last"),
                        ([question_range], [[last_token]], "Question -/-> Last"),
                        ([[last_token]], [[last_token]], "Last -/-> Last"),
                    ]

                elif args.target == "question-and-options-to-last":
                    block_mappings = [([question_without_options_range], [[last_token]], "Non-option question -/-> Last"),
                                      ([true_option_range], [[last_token]], "True option -/-> Last"),
                                      ([false_options_range], [[last_token]], "False option -/-> Last")]

                elif args.target == "vq-to-true-opt":
                    block_mappings = [
                        ([vision_range], [question_without_options_range], "Video -/-> Non-option question"),
                        ([question_without_options_range], [true_option_range], "Non-option question -/-> True option"),
                        ([vision_range], [true_option_range], "Video -/-> True option")]

                else:
                    raise NotImplementedError

                for block_key_range, block_query_range, block_desc in block_mappings:
                    # # **Precompute the attention mask once for this block configuration**
                    attn_mask = precompute_attention_masks(ntoks, mask_num_heads,
                                                           block_query_range, block_key_range, model.dtype,
                                                           model.device)

                    if args.sweep_range is not None:
                        sweep_range = tuple(map(int, args.sweep_range.split('-')))
                        sweep_layers = range(sweep_range[0], sweep_range[1])
                    else:
                        sweep_layers = range(num_layers)

                    for layer in sweep_layers:
                        if args.window_style == 'center':
                            layerlist = [l for l in
                                         range(max(0, layer - window // 2), min(num_layers, layer - (-window // 2)))]
                        elif args.window_style == 'inverse':
                            layerlist = [l for l in
                                         range(max(0, layer - window // 2), min(num_layers, layer - (-window // 2)))]
                            layerlist = [l for l in range(num_layers) if l not in layerlist]
                        elif args.window_style == 'top_down':  # (cur_layer, final)
                            layerlist = [l for l in range(layer, num_layers)]
                        elif args.window_style == 'bottom_up':
                            layerlist = [l for l in range(0, layer + 1)]  # (0, cur_layer)
                        else:
                            raise NotImplementedError

                        # **Pass only to selected layers**
                        if open_ended:
                            r_answer, r_gt, _, new_answer_t, new_output_text = generate_with_attn_block(
                                model, processor, inputs, conv, split_tag, answer_t, gt_t,
                                attn_mask, layerlist, model_type)
                        else:
                            r_answer, r_gt, _, new_answer_t = trace_with_attn_block(model, inputs, answer_t, gt_t,
                                                                                    attn_mask, layerlist, model_type)

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
                            "video_path": example['video_path'],
                            "data_id": data_idx,
                            "gt": gt,
                            "base_answer": answer,
                            "new_answer": new_answer
                        })
                        if open_ended:
                            results[-1]["gt_text"] = example["answer"]
                            results[-1]["base_output_text"] = output_text
                            results[-1]["new_output_text"] = new_output_text

                        # Show last result in tqdm without breaking the progress bar
                        tqdm.write(json.dumps(results[-1], indent=4))

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
        print(f"{acc_base=}, {acc_new=}")
        if args.eval_only:
            exit()

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


if __name__ == "__main__":
    main()

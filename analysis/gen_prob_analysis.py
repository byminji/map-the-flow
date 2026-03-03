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

from analysis.causal_intervention_tools import (decode_tokens, generate_from_input)


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generation probability analysis: Trace layer-wise answer probability change."
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

    parser.add_argument("--layer_norm", action='store_true',
                        help="Normalize before projection.")

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")

    parser.add_argument("--sample_mode", type=str, default="correct",
                        choices=["correct", "wrong", "all"],
                        help="Sample gathering mode. Default is to analyze with only correctly answered samples.")
    parser.add_argument("--eval_only", action='store_true',
                        help="Eval mode without probing")

    args = parser.parse_args()

    # Print args
    target = 'gen-prob-true-false-opt'
    model_path = args.model_path
    pooling_shape = tuple(map(int, args.pooling_shape.split('-')))
    print(f'{model_path=}, {pooling_shape=}')
    print(f'{args.conv_mode=}, {target=}')

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

    output_root = f'{args.output_dir}/{dataset_name}/{target}/{model_name}'
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

                "============= Candidates & GT ids ============="
                if open_ended:
                    gt_ids = processor.tokenizer(example["answer"], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                    gts = decode_tokens(processor.tokenizer, gt_ids)
                    for gt, gt_t in zip(gts, gt_ids):
                        if gt != "":    # non-empty first token
                            break

                else:
                    gt = example["answer"][1]  # e.g., 'A'
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                "============= Forward with hidden representation caching ============="
                answer_t, base_score, probs, output_text, hs = generate_from_input(model, processor, inputs, conv,
                                                                                   split_tag, return_hidden_states=True)

                true_opt, true_opt_t, false_opt, false_opt_t = [], [], [], []
                for candidate in example["candidates"]:
                    ids = processor.tokenizer(candidate, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                    cand = decode_tokens(processor.tokenizer, ids)
                    cand, cand_t = cand[0], ids[0]
                    try:
                        cand_t_2 = processor.tokenizer.get_vocab()[cand]
                        if abs(answer_t - cand_t_2) < abs(answer_t - cand_t):
                            cand_t = cand_t_2  # to avoid duplicated mappings
                    except:
                        pass

                    if cand == gt:
                        true_opt.append(cand)
                        true_opt_t.append(cand_t)
                    else:
                        false_opt.append(cand)
                        false_opt_t.append(cand_t)

                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                # get correct token probability
                base_score_gt = probs[gt_t].cpu().item()

                if args.sample_mode == "correct" and answer.lower() != gt.lower():
                    print("Skipping baseline wrong sample")
                    continue
                if args.sample_mode == "wrong" and answer.lower() == gt.lower():
                    print("Skipping baseline correct sample")
                    continue

                acc_base += 1 if answer.lower() == gt.lower() else 0
                cnt_samples += 1

                E = model.get_output_embeddings().weight.detach()
                norm = model.language_model.model.norm

                top_k_answers, top_k_scores = [], []
                for layer in range(num_layers):
                    # calculate logits with last input token
                    hs_cur = hs[layer]
                    if args.layer_norm and layer != num_layers - 1:
                        hs_cur = norm(hs_cur)
                    logits = hs_cur[0, -1, :].matmul(E.T)  # (vocab_size)
                    scores = torch.softmax(logits, dim=-1)  # (vocab_size)

                    # top-k prediction
                    probs_topk, preds_topk = torch.topk(scores, k=10, dim=-1)
                    top_k_answer = decode_tokens(processor.tokenizer, preds_topk)
                    top_k_answers.append(top_k_answer)
                    top_k_score = [x.cpu().item() * 100 for x in probs_topk]
                    top_k_scores.append(top_k_score)

                    # true option
                    for cls, t in zip(true_opt, true_opt_t):
                        results.append({
                            "prompt": prompt,
                            "layer": layer,
                            "desc": "True option",
                            "base_score": scores[answer_t].cpu().item() * 100,  # pred token score at each layer
                            "probability": scores[t].cpu().item() * 100,
                            "class": cls,
                            "video_path": example['video_path'],
                            "data_id": data_idx,
                            "gt": gt,
                            "base_answer": answer,
                            "preds_topk": top_k_answer,
                            "probs_topk": top_k_score,
                        })
                        if open_ended:
                            results[-1]["gt_text"] = example["answer"]
                            results[-1]["base_output_text"] = output_text

                    # false option
                    for cls, t in zip(false_opt, false_opt_t):
                        results.append({
                            "prompt": prompt,
                            "layer": layer,
                            "desc": "False option",
                            "base_score": scores[answer_t].cpu().item() * 100,  # pred token score at each layer
                            "probability": scores[t].cpu().item() * 100,
                            "class": cls,
                            "video_path": example['video_path'],
                            "data_id": data_idx,
                            "gt": gt,
                            "base_answer": answer,
                            "preds_topk": top_k_answer,
                            "probs_topk": top_k_score,
                        })
                        if open_ended:
                            results[-1]["gt_text"] = example["answer"]
                            results[-1]["base_output_text"] = output_text

                    # Show last result in tqdm without breaking the progress bar
                    tqdm.write(json.dumps(results[-1], indent=4))

                # Create a table for each position and layer
                print(f"{'Layer':<10}{'Top 1':<10}{'Top 2':<10}{'Top 3':<10}{'Top 4':<10}{'Top 5':<10}")
                for layer in range(num_layers):
                    print(f"{'Layer ' + str(layer):<10}", end="")
                    for token in top_k_answers[layer]:
                        print(f"{token.rstrip():<10}", end="")
                    print()  # Newline after the row
                print()

                pbar.update(1)  # Update progress after each QA sample

        "============= Visualization ============="

        def plot_results(results_data, results_keyword, y_data_name, hline_val=0):
            tmp = pd.DataFrame.from_records(results_data)
            tmp["layer_1"] = tmp.layer.apply(lambda x: x + 1)

            plt.figure(figsize=(8, 6))
            ax = sns.lineplot(tmp, x="layer_1", y=y_data_name,
                              hue="desc",
                              style="desc",
                              dashes=False,
                              palette=palette, linewidth=2)
            ax.set_xlabel("layer")
            ax.set_ylabel(f"% {y_data_name}")
            ax.set_xlim(0, num_layers + 0.5)
            sns.move_legend(ax, "lower right", title="class")
            plt.axhline(y=hline_val, color=palette[2], linestyle='-')

            plt.savefig(f"{output_root}/{results_keyword}_{y_data_name}_target_{target}_"
                        f"{task_i:02d}_{task_type}.png")

        acc_base = acc_base / cnt_samples * 100

        correct_results = [x for x in results if x["gt"].lower() == x["base_answer"].lower()]
        wrong_results = [x for x in results if x["gt"].lower() != x["base_answer"].lower()]
        plot_results(results, 'results', 'probability')
        if args.sample_mode == "all" and len(correct_results) > 0:
            plot_results(correct_results, 'correct_results', 'probability')
        if args.sample_mode == "all" and len(wrong_results) > 0:
            plot_results(wrong_results, 'wrong_results', 'probability')

        # Save results as a file
        os.makedirs(f"{output_root}/jsons", exist_ok=True)
        with open(f"{output_root}/jsons/{task_i:02d}_{task_type}.json", 'w') as f:
            json.dump(results, f, indent=4)

        print(f"{acc_base=}")


if __name__ == "__main__":
    main()

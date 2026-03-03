import argparse
import os
import datetime
import numpy as np
import torch.distributed as dist
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
from analysis.causal_intervention_tools import decode_tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generation probability analysis: Trace layer-wise answer probability change."
    )
    parser.add_argument("--model-path", "--model_path", type=str, default="workspace/models/VideoLLaMA3-7B")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--max-frames", "--max_frames", type=int, default=8)
    parser.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="workspace/gen_prob_analysis")
    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")

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
    target = 'gen-prob-true-false-opt'
    model_name = args.model_path.split('/')[-1]
    output_root = f'{args.output_dir}/{dataset_name}/{target}/{model_name}'
    print(f'{output_root=}')
    os.makedirs(output_root, exist_ok=True)

    # configurations
    num_layers = 28

    " ====== Arrange data ====== "
    # Arrange data by task type
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

                    "============= Forward with hidden representation caching ============="

                    @torch.no_grad()
                    def generate_from_input(model, processor, inputs):
                        output = model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                                output_scores=True, output_hidden_states=True,
                                                return_dict_in_generate=True)

                        scores = output.scores[0]  # first token score of generated answer
                        probs = torch.softmax(scores, dim=1)
                        p, preds = torch.max(probs, dim=1)

                        # output.hidden_states: tuple(gen_seq_len) of tuple(num_layer + 1) of tensor(batch_size, total_seq_len, dim)
                        # num_layer + 1 -> input (=input id embeddings) and output hidden representations at each layer
                        hs = output.hidden_states[0][1:]  # all layers' hidden representations at first generation step

                        # Decode into text
                        output_text = processor.batch_decode(output.sequences, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=False)[0]

                        return preds[0], p[0], probs[0], output_text, hs


                    answer_t, base_score, probs, output_text, hs = generate_from_input(model, processor, data_dict)

                    base_score = base_score.cpu().item()
                    [answer] = decode_tokens(processor.tokenizer, [answer_t])

                    gt = dataset.data_dict[data_id]['ground_truth']
                    if args.benchmark == 'tvbench':
                        gt = chr(gt + 65)
                    vocab = processor.tokenizer.get_vocab()
                    gt_t = vocab[gt]

                    base_score_gt = probs[gt_t].cpu().item()

                    if args.sample_mode == "correct" and answer.lower() != gt.lower():
                        print("Skipping baseline wrong sample")
                        continue
                    if args.sample_mode == "wrong" and answer.lower() == gt.lower():
                        print("Skipping baseline correct sample")
                        continue

                    true_opt, true_opt_t, false_opt, false_opt_t = [], [], [], []
                    for candidate in dataset.data_dict[data_id]['option_letters']:
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

                    acc_base += 1 if answer.lower() == gt.lower() else 0
                    cnt_samples += 1

                    input_ids = data_dict['input_ids'][0]
                    prompt_without_image_tokens = processor.batch_decode(input_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
                    prompt = ''.join(prompt_without_image_tokens)

                    E = model.get_output_embeddings().weight.detach()
                    norm = model.model.norm

                    top_k_answers, top_k_scores = [], []
                    for layer in range(num_layers):
                        # calculate logits with last input token
                        hs_cur = hs[layer]
                        if layer != num_layers - 1:
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
                                "data_id": data_id,
                                "gt": gt,
                                "base_answer": answer,
                                "preds_topk": top_k_answer,
                                "probs_topk": top_k_score,
                            })

                        # false option
                        for cls, t in zip(false_opt, false_opt_t):
                            results.append({
                                "prompt": prompt,
                                "layer": layer,
                                "desc": "False option",
                                "base_score": scores[answer_t].cpu().item() * 100,  # pred token score at each layer
                                "probability": scores[t].cpu().item() * 100,
                                "class": cls,
                                "data_id": data_id,
                                "gt": gt,
                                "base_answer": answer,
                                "preds_topk": top_k_answer,
                                "probs_topk": top_k_score,
                            })

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

            plt.savefig(f"{output_root}/{results_keyword}_{y_data_name}_target_{args.target}_"
                        f"{task_i:02d}_{task_type}.png")

            plt.show()
            plt.close()

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


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

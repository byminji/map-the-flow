import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from tqdm import tqdm
import json
torch.set_grad_enabled(False)
tqdm.pandas()
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.cm as cm
import pandas as pd
from collections import Counter
import random

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import decode_tokens, logit_lens_trace_with_proj


spatial_bag_of_words_dict = {
    '03_Action Sequence': [
        'bag', 'bed', 'blanket', 'book', 'box', 'cabinet', 'camera',
        'clothes', 'cup', 'glass',
        'bottle', 'dish', 'door', 'floor', 'food', 'glass',
        'laptop', 'paper', 'person', 'phone',
        'sandwich', 'table',
    ],
}
temporal_bag_of_words_dict = {
    '03_Action Sequence': [
        'eat', 'close', 'do', 'down', 'drink', 'hold', 'on', 'open',
        'put', 'sit', 'throw', 'tidy', 'take', 'up'
    ],
}


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Logit Lens analysis: Save layer-wise logits probing results from vision tokens. "
            "Results can be gathered and saved for all datasets, but visualization "
            "('--visualize_on_video' and '--visualize_frequency') is only supported for "
            "datasets/tasks with predefined bag-of-words (e.g., 'Action Sequence' in TVBench)."
        )
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

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")
    parser.add_argument("--test_id", type=int, default=-1,
                        help="Test sample id.")

    parser.add_argument("--sample_mode", type=str, default="correct",
                        choices=["correct", "wrong", "all"],
                        help="Sample gathering mode. Default is to analyze with only correctly answered samples.")

    parser.add_argument("--visualize_on_video", action='store_true',
                        help="Logit Lens visualization on video frames")
    parser.add_argument("--visualize_frequency", action='store_true',
                        help="Visualize frequency counts")

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
                                                       force_eager=True)

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
    for task_i, (task_type, qa_indices) in enumerate(video_index_map.items()):   # Iterate by task type
        if args.task_id != -1 and task_i != args.task_id:
            continue
        if args.test_ratio > 0:
            random.seed(42)
            qa_indices = random.sample(qa_indices, min(args.test_ratio, len(qa_indices)))
        if args.test_id != -1:
            qa_indices = [args.test_id]

        "============= Logit lens analysis ============="
        # Run attention knockouts
        acc_base, cnt_samples = 0, 0
        results = []
        with tqdm(total=len(qa_indices), desc=f"Processing QA Samples for {task_type}", unit="sample") as pbar:
            for i, data_idx in enumerate(qa_indices):
                example = dataset[data_idx]

                "============= Prepare inputs ============="
                # Prepare prompt
                video_list = example["video_pils"]  # list(frame_length) of PIL Image
                conv = conv_templates[conv_mode].copy()
                conv.user_query(example['question'], pre_query_prompt, post_query_prompt, is_mm=True)
                if answer_prompt is not None:
                    conv.assistant_response(answer_prompt)

                # Prepare inputs
                torch.cuda.empty_cache()
                prompt = conv.get_prompt()
                inputs = processor(text=prompt, images=video_list, return_tensors="pt").to(model.device)
                inputs['media_type'] = 'video'   # Needed for PLLaVA

                "============= Define token ranges ============="
                input_ids = inputs["input_ids"][0]
                image_placeholder_index = torch.where(input_ids == model.config.image_token_index)[0].item()
                num_vis = pooling_shape[0] * pooling_shape[1] * pooling_shape[2]
                vision_range = [x + image_placeholder_index for x in range(num_vis)]
                token_ranges = {
                    "video": vision_range
                }

                "============= Baseline forward without blocking ============="
                # prediction
                answer_t, base_score, projs, probs = logit_lens_trace_with_proj(model, inputs)
                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                # get correct token probability
                # if dataset_name == 'tvbench':
                gt = example["answer"][1]   # e.g., 'A'
                vocab = processor.tokenizer.get_vocab()
                gt_t = vocab[gt]

                base_score_gt = probs[gt_t].cpu().item()

                if args.sample_mode == "correct" and answer.lower() != gt.lower():
                    print("Skipping baseline wrong sample")
                    continue
                if args.sample_mode == "wrong" and answer.lower() == gt.lower():
                    print("Skipping baseline correct sample")
                    continue

                acc_base += 1 if answer.lower() == gt.lower() else 0
                cnt_samples += 1

                results.append({
                    "prompt": prompt,
                    "question": example['question'],
                    "pred_baseline": answer,
                    "pred_baseline_score": base_score,
                    "gt": gt,
                    "video_path": example['video_path'],
                    "data_id": data_idx,
                    "task_type": task_type
                })

                tqdm.write(json.dumps(results[-1], indent=4))

                for pos_name, pos in token_ranges.items():
                    results[-1][pos_name] = {}
                    for layer in range(num_layers):
                        preds = projs[f"layer_residual_{layer}_preds"]
                        answers = decode_tokens(processor.tokenizer, preds[0, pos, :5]) # list(len(pos)) of list(k)
                        results[-1][pos_name][layer] = answers

                    print(f"\nResults for position: {pos_name}")

                    if len(pos) == 1:
                        # Create a table for each position and layer
                        print(f"{'Layer':<10}{'Top 1':<10}{'Top 2':<10}{'Top 3':<10}{'Top 4':<10}{'Top 5':<10}")

                        # Print top-k predictions for each layer at the current position
                        for layer in range(num_layers):
                            top_k_predictions = results[-1][pos_name][layer][0]
                            print(f"{'Layer ' + str(layer):<10}", end="")
                            for token in top_k_predictions:
                                print(f"{token:<10}", end="")
                            print()  # Newline after the row
                    else:
                        print(f"{'Layer':<10}")
                        for layer in range(num_layers):
                            top_1_predictions = [x[0] for x in results[-1][pos_name][layer]]
                            print(f"{'Layer ' + str(layer):<10}", end="")
                            for token in top_1_predictions:
                                print(f"{token.strip():<10}", end="")
                            print()  # Newline after the row

                "============= Visualization on video frames in individual samples ============="
                if args.visualize_on_video:
                    task_name = f"{task_i:02d}_{task_type}"
                    save_root = f"{output_root}/{task_name}/{i:03d}_{data_idx:05d}"
                    os.makedirs(save_root, exist_ok=True)

                    def unflatten_position(pos, T, H, W):
                        t = pos // (H * W)
                        h = (pos % (H * W)) // W
                        w = pos % W
                        return t, h, w

                    def save_frame_images_multiple_layers(video_list, processor, token_info_dict, fig_name=None):
                        frames_without_norm = processor.preprocess_masks(masks=video_list,
                                                                         return_tensors="pt")['mask_values'].cuda()
                        token_T, token_H, token_W = 8, 12, 12

                        # grid with line
                        t, c, h, w = frames_without_norm.size()

                        assert t == token_T
                        assert h == w

                        num_rows = len(token_info_dict.keys())

                        frames_without_norm = rearrange(frames_without_norm, 't c h w -> h (t w) c')
                        frames_without_norm = frames_without_norm.repeat(num_rows, 1, 1)
                        frames_without_norm = frames_without_norm.cpu().numpy()
                        plt.figure(figsize=(16, 20))
                        plt.imshow(frames_without_norm)

                        num_frames = t
                        frame_borders = [k * w for k in range(num_frames + 1)]  # num_frames * w
                        layer_borders = [k * h for k in range(num_rows + 1)]  # num_layers * h

                        for idx in frame_borders:
                            plt.axvline(x=idx - 0.5, color='white', linestyle='-', linewidth=4)  # Vertical line

                        for idx in layer_borders:
                            plt.axhline(y=idx - 0.5, color='white', linestyle='-', linewidth=4)  # Horizontal line

                        # Remove x-axis and y-axis numbers
                        plt.xticks([])  # Remove x-axis numbers
                        plt.yticks([])  # Remove y-axis numbers

                        # Remove grid and borderlines
                        plt.grid(False)
                        plt.gca().spines['top'].set_visible(False)  # Remove top borderline
                        plt.gca().spines['right'].set_visible(False)  # Remove right borderline
                        plt.gca().spines['bottom'].set_visible(False)  # Remove bottom borderline
                        plt.gca().spines['left'].set_visible(False)  # Remove left borderline

                        for level, token_info in token_info_dict.items():
                            for word, color, pos in token_info:
                                # Convert to (t, h, w)
                                t_idx, h_idx, w_idx = unflatten_position(pos, token_T, token_H, token_W)
                                assert pos == t_idx * (token_H * token_W) + h_idx * (token_H) + w_idx

                                # Transform scale to image size
                                h_idx_scale = (h_idx + 0.5) * h / token_H  # add 0.5 for plt position
                                w_idx_scale = (w_idx + 0.5) * w / token_W  # add 0.5 for plt position

                                # Position translation to top-left
                                h_pos = h_idx_scale + level * h
                                w_pos = w_idx_scale + t_idx * w

                                plt.text(w_pos, h_pos, word, color=color, fontsize=12, ha='center', va='center',
                                         fontweight='bold')

                        # Make layout tight without borders
                        plt.tight_layout()

                        if fig_name is not None:
                            plt.savefig(fig_name)
                            print(f"Saved {fig_name}")

                        plt.show()
                        plt.close()

                    def get_first_tokenized_keywords(bag_of_words):
                        first_tokenized_keywords = []
                        for keyword in bag_of_words:
                            ids = processor.tokenizer(keyword, add_special_tokens=False)['input_ids']
                            first_tokenized_keywords.append(processor.tokenizer.decode(ids[0]))

                        first_tokenized_keywords = list(set(first_tokenized_keywords))
                        first_tokenized_keywords.sort()
                        return first_tokenized_keywords

                    spatial_vocab_first_token = get_first_tokenized_keywords(spatial_bag_of_words_dict[task_name])
                    temporal_vocab_first_token = get_first_tokenized_keywords(spatial_bag_of_words_dict[task_name])

                    def extract_vocabs(logit_lens_result, bag_of_words):
                        info = []
                        for layer_num, token_predictions in logit_lens_result.items():
                            for token_pos, top_5_list in enumerate(token_predictions):
                                top_1_prediction = top_5_list[0]
                                if top_1_prediction in bag_of_words:
                                    info.append(
                                        (top_1_prediction, int(layer_num), token_pos)
                                    )
                        return info

                    spatial_vocab_hits = extract_vocabs(results[-1][pos_name], spatial_vocab_first_token)
                    temporal_vocab_hits = extract_vocabs(results[-1][pos_name], temporal_vocab_first_token)

                    unique_temporal_vocab_in_sample = set(x[0] for x in spatial_vocab_hits)
                    unique_spatial_vocab_in_sample = set(x[0] for x in temporal_vocab_hits)

                    def generate_color_mapping(vocab):
                        color_mapping = {}
                        colormap = cm.get_cmap('tab20', len(vocab))
                        for i, word in enumerate(vocab):
                            color_mapping[word] = colormap(i)  # Map from 0 to 1
                        return color_mapping

                    color_mapping = generate_color_mapping(unique_temporal_vocab_in_sample | unique_spatial_vocab_in_sample)

                    layer_grid = 5
                    num_levels = (num_layers + layer_grid - 1) // layer_grid

                    #### multiple plots
                    # Temporal vocab
                    temporal_vocabs_to_visualize = {x: [] for x in range(num_levels)}
                    for word, layer, pos in temporal_vocab_hits:
                        temporal_vocabs_to_visualize[layer // layer_grid].append((word, color_mapping[word], pos))
                    save_frame_images_multiple_layers(video_list, processor, temporal_vocabs_to_visualize,
                                                      fig_name=f"{save_root}/temporal_vocab.png")

                    # Spatial vocab
                    spatial_vocabs_to_visualize = {x: [] for x in range(num_levels)}
                    for word, layer, pos in spatial_vocab_hits:
                        spatial_vocabs_to_visualize[layer // layer_grid].append((word, color_mapping[word], pos))
                    save_frame_images_multiple_layers(video_list, processor, spatial_vocabs_to_visualize,
                                                      fig_name=f"{save_root}/spatial_vocab.png")

                    # Spatial & temporal keyword at once
                    total_vocabs_to_visualize = {x: [] for x in range(num_levels)}
                    colormap = cm.get_cmap('tab20', 2)
                    for word, layer, pos in spatial_vocab_hits:
                        total_vocabs_to_visualize[layer // layer_grid].append((word, colormap(0), pos))
                    for word, layer, pos in temporal_vocab_hits:
                        total_vocabs_to_visualize[layer // layer_grid].append((word, colormap(1), pos))
                    save_frame_images_multiple_layers(video_list, processor, total_vocabs_to_visualize,
                                                      fig_name=f"{save_root}/all_vocab.png")

                pbar.update(1)  # Update progress after each QA sample

        # Save results as a file
        os.makedirs(f"{output_root}/jsons", exist_ok=True)
        task_name = f"{task_i:02d}_{task_type}"
        filename = f"{output_root}/jsons/{task_name}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

        "============= Visualize layerwise frequency ============="
        if args.visualize_frequency or task_name == '03_Action Sequence':
            pd.options.display.float_format = '{:.2f}'.format

            df = pd.read_json(filename)

            vocab_list = []
            for i in range(num_layers):
                vocab_counter = Counter()
                for item in df['video']:
                    top_1_preds = [x[0] for x in item[str(i)]]
                    vocab_counter.update(top_1_preds)
                vocab_list.append(vocab_counter)

            def draw_keywords_average(bag_of_words):
                first_tokenized_keywords = []
                for keyword in bag_of_words:
                    ids = processor.tokenizer(keyword, add_special_tokens=False)['input_ids']
                    first_tokenized_keywords.append(processor.tokenizer.decode(ids[0]))

                first_tokenized_keywords = list(set(first_tokenized_keywords))
                first_tokenized_keywords.sort()
                print(first_tokenized_keywords)

                # Initialize a dictionary to store counts for each keyword
                keyword_counts = {keyword: [] for keyword in first_tokenized_keywords}

                # Populate the dictionary with counts from vocab_list
                for keyword in first_tokenized_keywords:
                    for vocab_counter in vocab_list:
                        keyword_counts[keyword].append(vocab_counter[keyword])

                # Convert the dictionary to a DataFrame
                df_counts = pd.DataFrame(keyword_counts)

                # Transpose the DataFrame (swap rows and columns)
                df_swapped = df_counts.transpose()

                # Define ranges of layers (every 5 layers)
                layer_ranges = [(i, i + 4) for i in range(0, len(vocab_list), 5)]

                # Initialize a dictionary to store aggregated raw counts for each keyword and each layer range
                layer_range_raw_counts = {f'{start + 1}-{end + 1}': [] for start, end in layer_ranges}

                # Populate with raw counts (before normalization)
                for keyword in first_tokenized_keywords:
                    for start, end in layer_ranges:
                        # Sum raw counts across the range
                        range_raw_sum = df_swapped.loc[keyword, start:end + 1].sum()  # Use df_swapped (raw counts)
                        layer_range_raw_counts[f'{start + 1}-{end + 1}'].append(range_raw_sum)

                # Convert to DataFrame
                df_layer_raw = pd.DataFrame(layer_range_raw_counts, index=first_tokenized_keywords)

                # Remove rows where sum across all layers < 100
                df_layer_raw = df_layer_raw[df_layer_raw.sum(axis=1) >= 100]

                # Add ALL row (sum over all rows in each column)
                df_layer_raw.loc['ALL'] = df_layer_raw.sum(axis=0)

                # Now normalize each row to sum to 1
                df_layer_normalized = df_layer_raw.div(df_layer_raw.sum(axis=1), axis=0)
                df_layer_raw.loc['ALL_NORM'] = df_layer_normalized.drop('ALL').mean(axis=0)

                # normalize across keyrow
                return df_layer_normalized, df_layer_raw

            def plot_average(df_layer_normalized, save_path):
                # Plot the average normalized counts across layers
                df_average = df_layer_normalized.drop('ALL').mean(axis=0)
                plt.figure(figsize=(10, 6))
                plt.bar(df_average.index, df_average.values)
                plt.title("Average Normalized Counts Across Keywords")
                plt.xlabel("Layer")
                plt.ylabel("Average Normalized Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

            # Merge all lists into one without duplication
            all_spatial_words = []
            for task_words in spatial_bag_of_words_dict.values():
                all_spatial_words.extend(task_words)

            # Merge all lists into one without duplication
            all_temporal_words = []
            for task_words in temporal_bag_of_words_dict.values():
                all_temporal_words.extend(task_words)

            spatial_df_layer_normalized, spatial_df_layer_raw = draw_keywords_average(spatial_bag_of_words_dict[task_name])
            plot_average(spatial_df_layer_normalized)

            spatial_df_layer_raw.index = ['head(phone)' if idx == 'head' else idx for idx in spatial_df_layer_raw.index]
            spatial_df_layer_raw.index = ['sh(oes)' if idx == 'sh' else idx for idx in spatial_df_layer_raw.index]
            spatial_df_layer_raw.index = ['jack(et)' if idx == 'jack' else idx for idx in spatial_df_layer_raw.index]
            spatial_df_layer_raw.index = ['cy(an)' if idx == 'cy' else idx for idx in spatial_df_layer_raw.index]
            spatial_df_layer_raw.index = ['cyl(inder)' if idx == 'cyl' else idx for idx in spatial_df_layer_raw.index]
            spatial_df_layer_raw.index = ['pur(ple)' if idx == 'pur' else idx for idx in spatial_df_layer_raw.index]

            temporal_df_layer_normalized, temporal_df_layer_raw = draw_keywords_average(
                temporal_bag_of_words_dict[task_name])
            plot_average(temporal_df_layer_normalized)

            temporal_df_layer_raw.index = ['tid(y)' if idx == 'tid' else idx for idx in temporal_df_layer_raw.index]

            spatial_df_layer_raw.to_csv(f"{output_root}/jsons/{task_name}_logit_lens_spatial.csv")
            temporal_df_layer_raw.to_csv(f"{output_root}/jsons/{task_name}_logit_lens_temporal.csv")


if __name__ == "__main__":
    main()

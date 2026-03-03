import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import spacy
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
from analysis.causal_intervention_tools import (precompute_attention_masks, generate_with_attn_block,
                                                decode_tokens, generate_from_input, find_token_range)


def parse_list(value):
    return value.split('+')  # Split the input string by '+' and return as a list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Information flow analysis in open-ended answer generation: "
                    "Trace the probability change of N-th anchor generation after Attention Knockout."
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

    parser.add_argument("--generation_anchor_order", type=int, required=True,
                        help="Order of generation anchor to trace. Start from 1.")
    parser.add_argument("--target", type=str, default='vqrl-to-qrl',
                        help="Target blocking position.")
    parser.add_argument("--window", type=int, default=9,
                        help="Blocking window size. e.g., 9")
    parser.add_argument("--window_style", type=str, default='center',
                        choices=["center", "inverse", "top_down", "bottom_up"],
                        help="Blocking window style.")
    parser.add_argument("--sweep_range", type=str, default=None,
                        help="Layer sweeping start and end")
    parser.add_argument("--num_open_tokens", type=int, default=1)

    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task type index")
    parser.add_argument("--test_ratio", type=int, default=-1,
                        help="Test ratio. If given, randomly sample subset of the total dataset.")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens for generation")
    parser.add_argument("--scan_mode", action='store_true')

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
        model_name = model_name+"-"+args.generation_anchor_order

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
        results = []
        with (tqdm(total=len(qa_indices), desc=f"Processing QA Samples for {task_type}", unit="sample") as pbar):
            for i, data_idx in enumerate(qa_indices):
                example = dataset[data_idx]

                "============= Prepare inputs ============="
                # Prepare prompt
                video_list = example["video_pils"]  # list(frame_length) of PIL Image
                conv = conv_templates[conv_mode].copy()
                conv.user_query(example['question'], pre_query_prompt, post_query_prompt, is_mm=True)

                if answer_prompt is not None:
                    conv.assistant_response(answer_prompt)
                if conv.roles[-1] == "<|im_start|>assistant\n":
                    split_tag = "<|im_start|> assistant\n"
                else:
                    split_tag = conv.roles[-1]

                # Prepare inputs
                torch.cuda.empty_cache()
                prompt = conv.get_prompt()
                inputs = processor(text=prompt, images=video_list, return_tensors="pt").to(model.device)
                inputs['media_type'] = 'video'  # Needed for PLLaVA

                "============= Full forward ============="
                # full forward
                with torch.no_grad():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=256,
                                            num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0,
                                            length_penalty=1, temperature=1.0, output_scores=True,
                                            return_dict_in_generate=True)

                # First, gather full response without attention knockout
                output_text = processor.batch_decode(output.sequences, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
                full_output_text = output_text.split(split_tag)[-1]

                print(data_idx)
                print(prompt)
                print(full_output_text)
                print(example["answer"])

                if args.scan_mode:
                    pbar.update(1)
                    continue

                "============= Define token ranges ============="
                #### vision range
                # vision -> (<image> idx, <image> idx + num_vision_tokens)
                # because #num_vision_tokens tokens are inserted in the position of <image> token
                input_ids = inputs["input_ids"][0]
                image_placeholder_index = torch.where(input_ids == image_token_index)[0].item()
                num_vis = pooling_shape[0] * pooling_shape[1] * pooling_shape[2]
                vision_range = [x + image_placeholder_index for x in range(num_vis)]

                #### question range
                # question -> should shift by (num_vision_tokens - 1)
                question_range = find_token_range(processor.tokenizer, input_ids, example['question'])
                assert question_range[0] > -1
                question_token_ids = [input_ids[x].item() for x in range(question_range[0], question_range[1])]

                # assert reverse_check_token_range(processor.tokenizer, input_ids, question_range[0], question_range[1],
                #                                  example['question'])
                question_range = [x + (num_vis - 1) for x in range(question_range[0], question_range[1])]

                #### verb token range
                # Find all verbs in the response
                # We are tracking the generation of nth anchor (verb) when having (n-1) anchors
                # Here, n == args.generation_anchor_order
                nlp = spacy.load("en_core_web_lg")
                doc = nlp(full_output_text)
                pos_dict = {}
                r_ = [(token.text, token.pos_) for token in doc]
                for word, pos in r_:
                    if pos == "SPACE":
                        continue
                    if pos not in pos_dict:
                        pos_dict[pos] = []
                    pos_dict[pos].append(word)
                for pos in pos_dict:
                    pos_dict[pos] = sorted(set(pos_dict[pos]))
                if 'VERB' not in pos_dict.keys():
                    continue

                bag_of_verbs = pos_dict['VERB']

                input_len = len(input_ids)
                generated_tokens = output.sequences[0][input_len:]
                verb_pos = []   # track all token position
                verb_first_token_pos = []   # track only first tokenized position

                pos = 0
                decoded_words = decode_tokens(processor.tokenizer, generated_tokens)

                while pos < len(generated_tokens):
                    word = decoded_words[pos]
                    flag_verb = word in bag_of_verbs

                    # Case 1) can flag with single token
                    if flag_verb:
                        verb_pos.append(input_len + pos)
                        verb_first_token_pos.append(input_len + pos)
                        pos += 1
                        continue

                    # Case 2) can flag with two tokens
                    if pos + 1 < len(generated_tokens):
                        merged_word = decoded_words[pos] + decoded_words[pos + 1]
                        m_verb = merged_word in bag_of_verbs
                        if m_verb:
                            verb_pos.append(input_len + pos)
                            verb_pos.append(input_len + pos + 1)
                            verb_first_token_pos.append(input_len + pos)
                            pos += 2
                            continue

                    # Case 3) Not flagged -> move onto next position
                    pos += 1

                if len(verb_first_token_pos) < 3:    # We need samples with at least 3 verbs in response
                    continue
                pos = verb_first_token_pos[args.generation_anchor_order - 1]  # generation of nth vocab when having (n-1) vocabs
                target_token_id = output.sequences[0][pos]
                [last_token_text] = decode_tokens(processor.tokenizer, output.sequences[0][pos-1:pos])

                #### response range
                input_len = len(input_ids)

                target_position_at_input_ids = pos  # e.g., "speaking"
                last_response_position_at_input_ids = target_position_at_input_ids - 1  # e.g., "man"

                # shift with the number of visual tokens
                response_before_last_range = [x + (num_vis - 1) for x in
                                                range(input_len, last_response_position_at_input_ids)]
                last_token = last_response_position_at_input_ids + (num_vis - 1)

                cur_verb_pos = [x for x in verb_first_token_pos if x < last_response_position_at_input_ids]
                verb_anchors_text = decode_tokens(processor.tokenizer, output.sequences[0][cur_verb_pos])
                verb_range = [x + num_vis - 1 for x in cur_verb_pos]

                # exception for response before last range
                if args.num_open_tokens != 0:
                    # We do not block $num_open_tokens before the last token to avoid spike
                    if verb_range:
                        if verb_range[-1] in response_before_last_range:
                            last_verb_pos = response_before_last_range.index(verb_range[-1])
                            end_pos = max(last_verb_pos + 1, len(response_before_last_range) - args.num_open_tokens)
                        else:
                            end_pos = len(response_before_last_range) - args.num_open_tokens
                    else:
                        end_pos = len(response_before_last_range) - args.num_open_tokens
                    end_pos = max(1, end_pos)  # Ensure at least 1
                    open_len = len(response_before_last_range) - len(response_before_last_range[:end_pos])
                    response_before_last_range = response_before_last_range[:end_pos]

                if len(response_before_last_range) == 0:
                    continue

                non_last_token_range = question_range + response_before_last_range
                ntoks = last_token + 1

                if args.target == "vqrl-to-qrl":
                    block_mappings = [
                                      ([vision_range], [question_range], "Video -/-> Question"),
                                      ([vision_range], [response_before_last_range], "Video -/-> Non-last response"),
                                      ([vision_range], [[last_token]], "Video -/-> Last"),
                                      ([question_range], [[last_token]], "Question -/-> Last"),
                                      ([response_before_last_range], [[last_token]], "Non-last response -/-> Last"),
                                      ]
                else:
                    raise NotImplementedError

                "============= Base forward until anchor ============="
                #### change new input target
                # put prompt until anchor token (e.g., ...USER: What is the sequence of events in the video? ASSISTANT: The video shows a man"
                new_inputs = copy.deepcopy(inputs)
                new_inputs['input_ids'] = output.sequences[:, :target_position_at_input_ids]
                new_inputs['attention_mask'] = torch.ones_like(new_inputs.input_ids)

                answer_t, base_score, probs, output_text = generate_from_input(model, processor, new_inputs,
                                                                               conv, split_tag,
                                                                               max_new_tokens=args.max_new_tokens)

                base_score = base_score.cpu().item()
                [answer] = decode_tokens(processor.tokenizer, [answer_t])

                if answer_t != target_token_id:
                    continue

                "============= Attention Knockout ============="

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

                        r_answer, _, _, new_answer_t, new_output_text = generate_with_attn_block(
                            model, processor, new_inputs, conv, split_tag, answer_t, answer_t,
                            attn_mask, layerlist, model_type, max_new_tokens=args.max_new_tokens)

                        new_score = r_answer.cpu().item()
                        new_answer_t = new_answer_t.cpu().item()
                        [new_answer] = decode_tokens(processor.tokenizer, [new_answer_t])

                        results.append({
                            "prompt": prompt,
                            "block_desc": block_desc,
                            "layer": layer,
                            "base_score": base_score,
                            "new_score": new_score,
                            "relative_diff": (new_score - base_score) * 100.0 / base_score,
                            "video_path": example['video_path'],
                            "data_id": data_idx,
                            "last_token": last_token_text,
                            "verb_anchors": verb_anchors_text,
                            "base_answer": answer,
                            "new_answer": new_answer,
                            "base_output_text": output_text,
                            "new_output_text": new_output_text,
                            "response": full_output_text,
                            "ground_truth": example['answer'],
                        })

                        # Show last result in tqdm without breaking the progress bar
                        tqdm.write(json.dumps(results[-1], indent=4))

                        if len(layerlist) >= num_layers:
                            break

                pbar.update(1)  # Update progress after each QA sample

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

        plot_results(results, 'results', 'relative_diff')

        # Save results as a file
        os.makedirs(f"{output_root}/jsons", exist_ok=True)
        with open(f"{output_root}/jsons/{task_i:02d}_{task_type}.json",
                  'w') as f:
            json.dump(results, f, indent=4)

        print(f"Saved results in {output_root}/jsons/{task_i:02d}_{task_type}.json")


if __name__ == "__main__":
    main()

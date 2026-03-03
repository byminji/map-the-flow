import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter
from models.pllava import PllavaProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--output_root', type=str, required=True)
parser.add_argument('--filename', type=str, required=True)
args = parser.parse_args()

output_root = args.output_root
filename = args.filename

processor = PllavaProcessor.from_pretrained('workspace/models/llava-v1.6-vicuna-13b-hf')
num_layers=40

task_name = '03_Action Sequence'

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
plot_average(spatial_df_layer_normalized, f"{output_root}/spatial_average.png")

spatial_df_layer_raw.index = ['head(phone)' if idx == 'head' else idx for idx in spatial_df_layer_raw.index]
spatial_df_layer_raw.index = ['sh(oes)' if idx == 'sh' else idx for idx in spatial_df_layer_raw.index]
spatial_df_layer_raw.index = ['jack(et)' if idx == 'jack' else idx for idx in spatial_df_layer_raw.index]
spatial_df_layer_raw.index = ['cy(an)' if idx == 'cy' else idx for idx in spatial_df_layer_raw.index]
spatial_df_layer_raw.index = ['cyl(inder)' if idx == 'cyl' else idx for idx in spatial_df_layer_raw.index]
spatial_df_layer_raw.index = ['pur(ple)' if idx == 'pur' else idx for idx in spatial_df_layer_raw.index]

temporal_df_layer_normalized, temporal_df_layer_raw = draw_keywords_average(temporal_bag_of_words_dict[task_name])
plot_average(temporal_df_layer_normalized, f"{output_root}/temporal_average.png")

temporal_df_layer_raw.index = ['tid(y)' if idx == 'tid' else idx for idx in temporal_df_layer_raw.index]

os.makedirs(f"{output_root}/jsons", exist_ok=True)
spatial_df_layer_raw.to_csv(f"{output_root}/jsons/{task_name}_logit_lens_spatial.csv")
temporal_df_layer_raw.to_csv(f"{output_root}/jsons/{task_name}_logit_lens_temporal.csv")
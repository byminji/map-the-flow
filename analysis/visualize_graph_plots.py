import os
import pandas as pd
from tqdm import tqdm
import json
tqdm.pandas()

# Visuals
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import seaborn as sns

sns.set(context="notebook",
        rc={"font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
sns.set_theme(style='whitegrid')

palette_ = sns.color_palette("Set1")
palette = {
    'red': palette_[0],
    'blue': palette_[1],
    'green': palette_[2],
    'purple': palette_[3],
    'orange': palette_[4],
    'pink': palette_[7],
    'gray': palette_[8],
    'brown': palette_[6],
    "yellow": "#FFB302",
}


plt.rcParams['font.family'] = ['Times New Roman']
# plt.rcParams['font.family'] = ['Arial']


def gather_results(jsons_root='workspace/information_flow_analysis',
                   dataset='tvbench',
                   model='llava-next-7b-video-ft',
                   task_id='00_Action Antonym',
                   target_lists=('cross-frame', 'vql-to-ql'),
                   target_block_desc=("Video -/-> Question", "Video -/-> Last"),
                   correct_only=True,
                   key='block_desc'
                   ):

    results_all = []
    for target in target_lists:
        filename = f'{jsons_root}/{dataset}/{target}/{model}/{task_id}.json'

        with open(filename, 'r') as f:
            results = json.load(f)

        # pick correct results & desired lines only
        if correct_only:
            results = [x for x in results if x["gt"].lower() == x["base_answer"].lower()]

        results = [x for x in results if x[key] in target_block_desc]

        results_all.extend(results)

    return results_all


def plot_keyword_bar(df, keyword, palette, fig_name):
    """Plot a bar plot for a specific keyword and save it to a file."""
    # Get the data for the specific keyword (row in the DataFrame)
    keyword_data = df.loc[keyword]

    # Create a bar plot
    plt.figure(figsize=(8, 7))
    plt.bar(keyword_data.index, keyword_data.values, color=palette)

    # Set labels and title
    if keyword == 'ALL_NORM':
        title = 'All Average'
    else:
        title = f'"{keyword}"'
    plt.title(title, fontsize=50, y=1.05)
    plt.ylabel("Normalized Frequency", fontsize=36)
    plt.xlabel("Layer", fontsize=36)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Set the y-ticks interval to 0.1
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure if the figure name is provided
    if fig_name is not None:
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")

    # Show the plot
    plt.show()
    plt.close()


def plot_results(results_data, palette, hline_val=0, num_layers=32, fig_name=None, dashes=False,
                 legend=False, yticks_interval=None, ylim=None,
                 y_name="relative_diff",
                 y_label="% Change in Probability",
                 keyword="block_desc",
                 figsize=(8, 7),
                 x_fontsize=42,
                 y_fontsize=40,
                 alpha=None
                 ):
    tmp = pd.DataFrame.from_records(results_data)
    tmp["layer_1"] = tmp.layer.apply(lambda x: x + 1)

    plt.figure(figsize=figsize)
    if alpha is not None:
        ax = sns.lineplot(tmp, x="layer_1", y=y_name,
                          hue=keyword,
                          style=keyword,
                          dashes=dashes,
                          palette=palette,
                          linewidth=3,
                          err_kws={"alpha": alpha}
                          )
    else:
        ax = sns.lineplot(tmp, x="layer_1", y=y_name,
                          hue=keyword,
                          style=keyword,
                          dashes=dashes,
                          palette=palette,
                          linewidth=3,
                          )

    # Set labels with larger fonts
    ax.set_xlabel("Layer", fontsize=x_fontsize)
    ax.set_ylabel(y_label, fontsize=y_fontsize)

    # Set custom x-axis ticks
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)

    ax.set_xlim(1, num_layers)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    if ylim is not None:
        ax.set_ylim(ylim)

    if yticks_interval is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks_interval))

    if legend:
        sns.move_legend(ax, "lower right", title="blocked positions")
    else:
        ax.get_legend().remove()

    plt.axhline(y=hline_val, linestyle='-', color='gray')

    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")

    plt.show()
    plt.close()


def plot_results_for_teaser(results_data, palette, hline_val=0, num_layers=32, fig_name=None, dashes=False,
                 legend=False, yticks_interval=None, ylim=None,
                 y_name="relative_diff",
                 y_label="% Change in Probability",
                 keyword="block_desc"
                 ):
    tmp = pd.DataFrame.from_records(results_data)
    tmp["layer_1"] = tmp.layer.apply(lambda x: x + 1)

    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(tmp, x="layer_1", y=y_name,
                      hue=keyword,
                      style=keyword,
                      dashes=dashes,
                      palette=palette, linewidth=3)
    # Set labels with larger fonts
    ax.set_xlabel("Layer", fontsize=28)
    ax.set_ylabel(y_label, fontsize=28)

    # Set custom x-axis ticks
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    ax.set_xlim(1, 32)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    if ylim is not None:
        ax.set_ylim(ylim)

    if yticks_interval is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks_interval))

    if legend:
        sns.move_legend(ax, "lower right", title="blocked positions")
    else:
        ax.get_legend().remove()

    plt.axhline(y=hline_val, linestyle='-', color='gray')

    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")

    plt.show()
    plt.close()


def create_legend(legend_items, filename, linestyle=None, figsize=None, ncol=None, bar_plot=False, vertical=False):
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else ((10, 3) if vertical else (30, 1)))
    handles = []
    labels = []

    for i, (legend_name, color) in enumerate(legend_items):
        linestyle_cur = linestyle[i] if linestyle is not None else "-"

        if bar_plot:
            handles.append(Patch(color=color, label=legend_name))
        else:
            handles.append(
                plt.Line2D([0, 1], [0, 1], color=color, linewidth=5, linestyle=linestyle_cur)
            )
        labels.append(legend_name)

    if ncol is None:
        ncol = 1 if vertical else len(legend_items)

    ax.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=40)
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", transparent=True)
    plt.show()
    plt.close()


def plot_cross_frame_interaction_flow(jsons_root, save_root, baseline_name, video_ft_name):
    dataset = 'tvbench'

    target_block_desc = ['No cross-frame interactions']
    target_name = "cross-frame"

    baseline_name_small = baseline_name.lower()
    video_ft_name_small = video_ft_name.lower()

    num_layers = 40 if "13B" in baseline_name else 32

    save_model_name = f"{baseline_name_small}-vs-{video_ft_name_small}"

    palette_here = [palette['green'],
                    palette['pink']
                    ]

    tasks_list = [
        ['00_Action Antonym', None, None],
        ['03_Action Sequence', None, None],
        ['08_Scene Transition', None, None],
        ['05_Moving Direction', None, None],
        ['06_Object Count', None, None]
    ]
    if baseline_name == "LLaVA-NeXT-7B":
        tasks_list = [
            ['00_Action Antonym', 5, (-16, 3)],
            ['03_Action Sequence', 5, (-16, 3)],
            ['08_Scene Transition', 5, (-12, 3)],
            ['05_Moving Direction', 10, None],
            ['06_Object Count', None, None]
        ]

    # Save legend
    legend_items = [
        (f'No cross-frame interactions ({video_ft_name})', palette_here[0]),
        (f'No cross-frame interactions ({baseline_name})', palette_here[1])
    ]
    fig_name = f"{save_root}/{dataset}/{target_name}/{save_model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name)

    for task_id, yticks_interval, ylim in tasks_list:
        results_video_ft = gather_results(jsons_root, dataset, video_ft_name_small,
                                                task_id, [target_name], target_block_desc)
        results_baseline = gather_results(jsons_root, dataset, baseline_name_small,
                                       task_id, [target_name], target_block_desc)

        for x in results_video_ft:
            x['block_desc'] = legend_items[0][0]
        for x in results_baseline:
            x['block_desc'] = legend_items[1][0]

        results_all = results_video_ft + results_baseline

        fig_name = f"{save_root}/{dataset}/{target_name}/{save_model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval,
                     ylim=ylim, num_layers=num_layers)


def plot_vql_to_ql_flow(jsons_root, save_root, model_name, dataset='tvbench', figsize=(8, 7),
                        legend_vertical=False, x_fontsize=42, y_fontsize=40):

    target_name = "vql-to-ql"
    target_block_desc = ['Video -/-> Question',
                         'Video -/-> Last',
                         'Question -/-> Last',
                         'Last -/-> Last'
                         ]

    if model_name == 'llava-next-13b-video-ft':
        num_layers = 40
    elif model_name == 'videollama3-7b':
        num_layers = 28
    else:
        num_layers = 32

    palette_here = [palette['purple'],
                    palette['blue'],
                    palette['orange'],
                    palette['red']]

    if dataset == 'tvbench':
        tasks_list = [
            ['00_Action Antonym', None, None],
            ['03_Action Sequence', None, None],
            ['08_Scene Transition', None, None],
            ['05_Moving Direction', None, None],
            ['06_Object Count', None, None]
        ]
    elif dataset in ['longvideobench', 'longvideobench-8frame']:
        tasks_list = [
            ['02_O2E', None, None],
            ['03_O3O', None, None],
            ['08_SOS', None, None],
        ]
    elif dataset == 'videomme':
        tasks_list = [
            ['01_Action Recognition', None, None],
            ['08_Spatial Perception', None, None],
        ]

    # Save legend
    legend_items = [
        ('Video $\\nrightarrow$ Question', palette_here[0]),
        ('Video $\\nrightarrow$ Last', palette_here[1]),
        ('Question $\\nrightarrow$ Last', palette_here[2]),
        ('Last $\\nrightarrow$ Last', palette_here[3])
    ]

    fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name, vertical=legend_vertical)

    for task_id, yticks_interval, ylim in tasks_list:
        results_all = gather_results(jsons_root, dataset, model_name, task_id, [target_name], target_block_desc)

        fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval,
                     ylim=ylim, num_layers=num_layers, figsize=figsize, x_fontsize=x_fontsize, y_fontsize=y_fontsize)


def plot_q_option_to_last(jsons_root, save_root, model_name, dataset='tvbench', figsize=(8, 7),
                          legend_vertical=False, x_fontsize=42, y_fontsize=40):

    target_block_desc = ['Non-option question -/-> Last',
                         'True option -/-> Last',
                         'False option -/-> Last'
                         ]

    target_name = "question-and-options-to-last"

    if model_name == 'llava-next-13b-video-ft':
        num_layers = 40
    elif model_name == 'videollama3-7b':
        num_layers = 28
    else:
        num_layers = 32

    palette_here = [palette['red'],
                    palette['orange'],
                    palette['blue']
                    ]

    if dataset == 'tvbench':
        tasks_list = [
            ['00_Action Antonym', None, None],
            ['03_Action Sequence', None, None],
            ['08_Scene Transition', None, None],
            ['05_Moving Direction', None, None],
            ['06_Object Count', None, None]
        ]
        if model_name == 'llava-next-13b-video-ft':
            tasks_list = [
                ['00_Action Antonym', None, None],
                ['03_Action Sequence', None, None],
                ['08_Scene Transition', None, None],
                ['05_Moving Direction', None, None],
                ['06_Object Count', 2, None]
            ]
    elif dataset == 'tvbench-fail':
        tasks_list = [
            ['00_Action Antonym', None, None],
            ['05_Moving Direction', None, None],
        ]
    elif dataset in ['longvideobench', 'longvideobench-8frame']:
        tasks_list = [
            ['02_O2E', None, None],
            ['03_O3O', None, None],
            ['08_SOS', None, None],
        ]
    elif dataset == 'videomme':
        tasks_list = [
            ['01_Action Recognition', None, None],
            ['08_Spatial Perception', None, None],
        ]

    # Save legend
    legend_items = [
        ('Non-option question $\\nrightarrow$ Last', palette_here[0]),
        ('True option $\\nrightarrow$ Last', palette_here[1]),
        ('False option $\\nrightarrow$ Last', palette_here[2]),
    ]

    fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name, vertical=legend_vertical)

    for task_id, yticks_interval, ylim in tasks_list:
        if dataset == 'tvbench-fail':
            results_all = gather_results(jsons_root, dataset, model_name, task_id, [target_name], target_block_desc, correct_only=False)
        else:
            results_all = gather_results(jsons_root, dataset, model_name, task_id, [target_name], target_block_desc)

        fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval,
                     ylim=ylim, num_layers=num_layers, figsize=figsize, x_fontsize=x_fontsize, y_fontsize=y_fontsize)


def plot_vision_question_to_true_option(jsons_root, save_root, model_name, dataset='tvbench', figsize=(8, 7),
                                        legend_vertical=False, x_fontsize=42, y_fontsize=40):

    target_name = "vq-to-true-opt"

    target_block_desc = ['Video -/-> Non-option question',
                         'Non-option question -/-> True option',
                         'Video -/-> True option'
                         ]

    if model_name == 'llava-next-13b-video-ft':
        num_layers = 40
    elif model_name == 'videollama3-7b':
        num_layers = 28
    else:
        num_layers = 32

    palette_here = [palette['red'],
                    palette['red'],
                    palette['purple']
                    ]

    if dataset == 'tvbench':
        tasks_list = [
            ['00_Action Antonym', None, None],
            ['03_Action Sequence', None, None],
            ['08_Scene Transition', None, None],
            ['05_Moving Direction', None, None],
            ['06_Object Count', None, None]
        ]
    elif dataset in ['longvideobench', 'longvideobench-8frame']:
        tasks_list = [
            ['02_O2E', None, None],
            ['03_O3O', None, None],
            ['08_SOS', None, None],
        ]
    elif dataset == 'videomme':
        tasks_list = [
            ['01_Action Recognition', None, None],
            ['08_Spatial Perception', None, None],
        ]

    linestyle = ['-', '--', ':']

    # Save legend
    legend_items = [
        ('Video $\\nrightarrow$ Non-option question', palette_here[0]),
        ('Non-option question $\\nrightarrow$ True option', palette_here[1]),
        ('Video $\\nrightarrow$ True option', palette_here[2]),
    ]

    fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name, linestyle=linestyle, vertical=legend_vertical)

    for task_id, yticks_interval, ylim in tasks_list:
        results_all = gather_results(jsons_root, dataset, model_name, task_id, [target_name], target_block_desc)

        fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval,
                     ylim=ylim, num_layers=num_layers, dashes=True,
                     figsize=figsize, x_fontsize=x_fontsize, y_fontsize=y_fontsize)


def plot_vision_question_to_false_option(jsons_root, save_root, model_name, dataset='tvbench-fail', figsize=(8, 7),
                                         legend_vertical=False, x_fontsize=42, y_fontsize=40):
    target_name = "vq-to-false-opt"

    target_block_desc = ['Video -/-> Non-option question',
                         'Non-option question -/-> False option',
                         'Video -/-> False option'
                         ]

    if model_name == 'llava-next-13b-video-ft':
        num_layers = 40
    elif model_name == 'videollama3-7b':
        num_layers = 28
    else:
        num_layers = 32

    palette_here = [palette['red'],
                    palette['red'],
                    palette['purple']
                    ]

    tasks_list = [
        ['00_Action Antonym', None, None],
        ['05_Moving Direction', None, None],
    ]

    linestyle = ['-', '--', ':']

    # Save legend
    legend_items = [
        ('Video $\\nrightarrow$ Non-option question', palette_here[0]),
        ('Non-option question $\\nrightarrow$ False option', palette_here[1]),
        ('Video $\\nrightarrow$ False option', palette_here[2]),
    ]

    fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name, linestyle=linestyle, vertical=legend_vertical)

    for task_id, yticks_interval, ylim in tasks_list:
        results_all = gather_results(jsons_root, dataset, model_name, task_id, [target_name], target_block_desc,
                                     correct_only=False)

        fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval,
                     ylim=ylim, num_layers=num_layers, dashes=True, figsize=figsize,
                     x_fontsize=x_fontsize, y_fontsize=y_fontsize)


def plot_gen_prob(jsons_root, save_root, model_name):
    dataset = 'tvbench'

    target_desc = ['True option', 'False option']

    save_target_name = "gen-prob-true-false-opt"

    if model_name == 'llava-next-13b-video-ft':
        num_layers = 40
    elif model_name == 'videollama3-7b':
        num_layers = 28
    else:
        num_layers = 32

    palette_here = [palette['yellow'],
                    palette['blue']]

    tasks_list = [
        ['00_Action Antonym', None, None],
        ['03_Action Sequence', None, None],
        ['08_Scene Transition', None, None],
        ['05_Moving Direction', None, None],
        ['06_Object Count', None, None]
    ]

    # Save legend
    legend_items = [
        ('True option', palette_here[0]),
        ('False option', palette_here[1]),
    ]

    fig_name = f"{save_root}/{dataset}/{save_target_name}/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name)

    for task_id, yticks_interval, ylim in tasks_list:
        results_all = gather_results(jsons_root, dataset, model_name, task_id, [save_target_name], target_desc, key='desc')

        fig_name = f"{save_root}/{dataset}/{save_target_name}/{model_name}/{task_id.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, yticks_interval=yticks_interval, ylim=ylim,
                     y_name='probability',
                     y_label="% Probability",
                     keyword="desc",
                     num_layers=num_layers
                     )


def plot_logit_lens(jsons_root, save_root, model_name='llava-next-13b-video-ft'):
    """Read the CSV, plot each keyword as a bar plot, and save it."""
    dataset = 'tvbench'
    task_list = ['03_Action Sequence']

    palette_here = ["#CCCCCC", "#9DACBB", "#6E8DAB", "#3F6D9B", "#104E8B"]

    target_list = [
        ["logit-lens-spatial", palette_here[1]],
        ["logit-lens-temporal-clean", palette_here[2]],
    ]

    legend_items = [
        ('Spatial Keywords', palette_here[1]),
        ('Temporal Keywords', palette_here[2]),
    ]
    fig_name = f"{save_root}/{dataset}/logit-lens-spatial/{model_name}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name, bar_plot=True)

    for task_id in task_list:
        for target_name, color in target_list:
            # Construct the CSV file path
            file_path = f'{jsons_root}/{dataset}/{target_name}/{model_name}/{task_id}.csv'

            # Read the CSV into a DataFrame
            df = pd.read_csv(file_path, index_col=0)  # Assuming first column is index (keyword)

            # sum to one
            df = df.div(df.sum(axis=1), axis=0)

            # Ensure the save directory exists
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            # Plot a bar chart for each keyword and save it
            for keyword in df.index:
                # Define the path to save the plot
                fig_name = f"{save_root}/{dataset}/{target_name}/{model_name}/{task_id}_{keyword}.png"
                os.makedirs(os.path.dirname(fig_name), exist_ok=True)

                # Plot and save the bar plot
                plot_keyword_bar(df, keyword, color, fig_name)


def open_ended_vcgbench(json_root, save_root):
    dataset = 'vcgbench'

    df_one = pd.read_json(f"{json_root}/{dataset}/vqrl-to-qrl/llava-next-7b-video-ft-num-vocab-1/02_temporal_qa.json")
    df_two = pd.read_json(f"{json_root}/{dataset}/vqrl-to-qrl/llava-next-7b-video-ft-num-vocab-2/02_temporal_qa.json")
    df_three = pd.read_json(f"{json_root}/{dataset}/vqrl-to-qrl/llava-next-7b-video-ft-num-vocab-3/02_temporal_qa.json")

    ids_one = set(df_one["data_id"])
    ids_two = set(df_two["data_id"])
    ids_three = set(df_three["data_id"])
    common_ids = ids_one & ids_two & ids_three
    # filter only common set
    df_one = df_one[df_one["data_id"].isin(common_ids)].reset_index(drop=True)
    df_two = df_two[df_two["data_id"].isin(common_ids)].reset_index(drop=True)
    df_three = df_three[df_three["data_id"].isin(common_ids)].reset_index(drop=True)
    results_one = df_one.to_dict("records")
    results_two = df_two.to_dict("records")
    results_three = df_three.to_dict("records")

    target_block_desc = ['Video -/-> Question',
                         'Question -/-> Last',
                         'Video -/-> Last',
                         'Video -/-> Non-last response',
                         'Non-last response -/-> Last'
                         ]
    target_names = ['video-to-question',
                    'question-to-last',
                    'video-to-last',
                    'video-to-response',
                         'response-to-last'
                         ]

    palette_here = [palette['red'],
                    palette['orange'],
                    palette['blue'],
                    ]

    legend_items = [
        (f"First anchor generation", palette_here[0]),
        (f"Second anchor generation", palette_here[1]),
        (f"Third anchor generation", palette_here[2]),
    ]
    fig_name = f"{save_root}/{dataset}/legend.png"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    create_legend(legend_items, fig_name)

    # Save legend
    for target_block, target_name in zip(target_block_desc,target_names):
        cur_results_one = [x for x in results_one if x['block_desc'] == target_block]
        cur_results_two = [x for x in results_two if x['block_desc'] == target_block]
        cur_results_three = [x for x in results_three if x['block_desc'] == target_block]

        for x in cur_results_one:
            x['block_desc'] = legend_items[0][0]
        for x in cur_results_two:
            x['block_desc'] = legend_items[1][0]
        for x in cur_results_three:
            x['block_desc'] = legend_items[2][0]

        def filter_negative_only(data_list):
            df = pd.DataFrame(data_list)
            # Group by 'data_id' and compute the sum of 'relative_diff' for each group
            grouped_df = df.groupby('data_id').agg({'relative_diff': 'mean'}).reset_index()
            # Separate into two groups based on the sum of 'relative_diff'
            group_negative = grouped_df[grouped_df['relative_diff'] <= 0]
            # Now, merge these back with the original data to get the full details
            negative_data = pd.merge(group_negative, df, on='data_id', how='inner')
            unique_negative_data_ids = negative_data['data_id'].unique()
            # Gather data points for positive data
            df_filtered = df[df['data_id'].isin(unique_negative_data_ids)]
            return df_filtered.to_dict(orient='records')

        cur_results_one = filter_negative_only(cur_results_one)
        cur_results_two = filter_negative_only(cur_results_two)
        cur_results_three = filter_negative_only(cur_results_three)

        results_all = cur_results_one + cur_results_two + cur_results_three

        fig_name = f"{save_root}/{dataset}/{target_name}.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)

        plot_results(results_all, palette_here, fig_name=fig_name, num_layers=32,
                     y_name="relative_diff", figsize=(9, 7), alpha=0.12)


def main():
    jsons_root = 'path/to/precomputed/jsons'
    save_root = 'path/to/pngs/to/be/saved'

    ######## information flow analysis (e.g., llava-next-7b-video-ft)
    # cross-frame interactions
    plot_cross_frame_interaction_flow(jsons_root, save_root,
                                      baseline_name="LLaVA-NeXT-7B", video_ft_name="LLaVA-NeXT-7B-Video-FT")
    # vision to question & last
    plot_vql_to_ql_flow(jsons_root, save_root, model_name="llava-next-7b-video-ft")
    # question, true option, false option to last
    plot_q_option_to_last(jsons_root, save_root, model_name="llava-next-7b-video-ft")
    # vision, question to true option
    plot_vision_question_to_true_option(jsons_root, save_root, model_name="llava-next-7b-video-ft")
    # generation probability
    plot_gen_prob(jsons_root, save_root, model_name="llava-next-7b-video-ft")

    ######## logit lens frequency counts
    plot_logit_lens(jsons_root, save_root)

    ######## open-ended plots (vcgbench)
    open_ended_vcgbench(jsons_root, save_root)


if __name__ == "__main__":
    main()

## Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

> [**[ICLR 2026] Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs**](https://arxiv.org/abs/2510.13251) \
> [Minji Kim*](https://byminji.github.io), [Taekyung Kim*](https://scholar.google.co.kr/citations?user=u-9bdkwAAAAJ&hl=en), [Bohyung Han](https://cv.snu.ac.kr/index.php/~bhhan/) <br>
<sub> (* Equal Contribution) <br>

[![website](https://img.shields.io/badge/Project_Page-Map_the_Flow-yellow.svg)](https://map-the-flow.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.13251-b31b1b.svg)](https://arxiv.org/abs/2510.13251)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR-red.svg)](https://openreview.net/forum?id=QCB0HN61TU)
[![Hugging Face](https://img.shields.io/badge/-HuggingFace-3B4252?logo=huggingface&style=flat)](https://huggingface.co/collections/byminji/map-the-flow)


Official PyTorch implementation of the ICLR 2026 paper "**Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs**"

### Updates
* **2026/03/03**: Code and models released.
* **2026/01/26**: Our paper is accepted to ICLR 2026 with strong reviews! 🎉


## Overview

![teaser](docs/teaser.jpg)

**TL;DR**: This paper presents a systematic analysis of where and how information flows in VideoLLMs for temporal reasoning in VideoQA, revealing key patterns and effective pathways.


📍 **Summary of our findings on VideoLLMs' information flow**:

(a) Temporal reasoning begins with cross-frame interactions within video tokens at early-middle layers ![green](https://img.shields.io/badge/-green-4DAF4A),
followed by video-language integration into temporal keywords in the question ![purple](https://img.shields.io/badge/-purple-984EA3).
This information is conveyed to the last token at middle-late layers ![orange](https://img.shields.io/badge/-orange-FF7F00),
where answer generation occurs ![yellow](https://img.shields.io/badge/-yellow-FFB302).

(b) These effective pathways are identified via Attention Knockout, which disconnects attention pairs and tracks the drop in probability of the final answer to quantify their impact.

(c) Layer-wise answer probability rises immediately after video-language integration, indicating that the model is ready to predict correct answers after the middle layers.

Based on our analysis, we show that VideoLLMs can retain their VideoQA performance by selecting **effective information pathways** while **suppressing a substantial amount of attention edges**, e.g., **58%** in LLaVA-NeXT-7B-Video-FT.


📍 **This repository supports:**
- Causal intervention tools for VideoLLMs (e.g., Attention Knockout, Logit Lens, Attention Map Visualization)
- Reproducible experiments from our paper, including figure plotting code
- Training and evaluation across various model series and video benchmarks


## Models

You can download all model checkpoints from the Hugging Face links below.
We fine-tuned LLaVA-NeXT and Mini-InternVL on [VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) to analyze the impact of video instruction tuning on model behavior.
We also adopted VideoLLaMA3 without additional fine-tuning.

| Model                     | Link                                                                                                                                                           | Initialized From                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| LLaVA-NeXT-7B-Video-FT    | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/byminji/LLaVA-NeXT-7B-Video-FT)    | [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)           |
| LLaVA-NeXT-13B-Video-FT   | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/byminji/LLaVA-NeXT-13B-Video-FT)   | [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)         |
| Mini-InternVL-4B-Video-FT | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/byminji/Mini-InternVL-4B-Video-FT) | [OpenGVLab/Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)  |
| VideoLLaMA3-7B            | -                                                                                                                                                              | [DAMO-NLP-SG/VideoLLaMA3-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)                       |


## Environments

### Installation

Tested with Python 3.10, PyTorch 2.2.1, CUDA 11.8. Other versions may be compatible.

Step 1: Create a virtual environment

- Option 1: PyTorch Docker image with torch==2.2.1, torchaudio==2.2.1, torchvision==0.17.1
  ```bash
  docker run -it --gpus all --ipc=host --rm --name=map_the_flow \
  pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
  ```

- Option 2: Conda environment
  ```bash
  conda create -n map_the_flow python=3.10 -y
  conda activate map_the_flow
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  pytorch-cuda=11.8 -c pytorch -c nvidia -y
  ```

Step 2: Clone the repository and install dependencies
```bash
git clone https://github.com/byminji/map-the-flow.git
cd map-the-flow

pip install -r requirements.txt
pip install mmcv-full==1.7.2 --no-build-isolation # mmcv-full must be built from source
```


### Data preparation

You can download all evaluation data from the Hugging Face links below.
After downloading, set the paths in [tasks/eval/config_dataset.py](tasks/eval/config_dataset.py).

- [TVBench](https://huggingface.co/datasets/FunAILab/TVBench): We mainly adopted TVBench for our analysis.
- [TOMATO](https://huggingface.co/datasets/yale-nlp/TOMATO): Adopted for effective pathway analysis.
- [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench): Long video understanding analysis.
- [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME): Spatial understanding analysis.
- [VCGBench](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main): Used for open-ended analysis. We followed the original [repo](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main) to prepare the evaluation data.


## Analysis

All implementations are in [analysis](analysis) folder and run scripts are in [scripts/analysis](scripts/analysis).
Results including graph plots and raw data are saved under `${output_path}/${dataset_name}/${target}/${model_name}`.
To reproduce the plot style used in our paper, run [analysis/visualize_graph_plots.py](analysis/visualize_graph_plots.py) on the saved JSONs.


### Common variables

Modify these variables at the top of each script before running.

| Variable | Description                          | Example |
|----------|--------------------------------------|---------|
| `dataset_name` | Evaluation dataset                   | `tvbench` |
| `output_path` | Root directory for saving results    | `workspace/outputs/information_flow_analysis` |
| `video_model_path` | Path to the fine-tuned model         | `workspace/models/LLaVA-NeXT-7B-Video-FT` |
| `base_model_path` | Path to the base model               | `workspace/models/llava-v1.6-vicuna-7b-hf` |
| `conv_mode` | Conversation template                | `eval_mvbench` |
| `pooling_shape` | Token pooling shape (`T-H-W`)        | `8-12-12` |
| `task_id` | Task index (`-1` = full dataset) | `0` |

Task IDs for TVBench: `0`=Action Antonym, `3`=Action Sequence, `5`=Moving Direction, `6`=Object Count, `8`=Scene Transition.


### Information flow analysis

- Casually traces the impact of specific token interactions by using Attention Knockout.
- Script: [scripts/analysis/information_flow_analysis_*.sh](scripts/analysis)
- Implementation: [analysis/information_flow_analysis.py](analysis/information_flow_analysis.py)

| `--target` | Description                                           |
|------------|-------------------------------------------------------|
| `cross-frame` | Block cross-frame interactions among video tokens     |
| `vql-to-ql` | Block video/question/last → question/last flows       |
| `question-and-options-to-last` | Block question-only, true, false options → last token |
| `vq-to-true-opt` | Block video/question → true option token              |


### Generation probability analysis

- Traces layer-wise answer probability changes for true/false options (Fig. 9 in our paper).
- Script: included at the end of [scripts/analysis/information_flow_analysis_*.sh](scripts/analysis)
- Implementation: [analysis/gen_prob_analysis.py](analysis/gen_prob_analysis.py)


### Effective pathway analysis

- Disconnects attentions except for those idefined as effective pathways, showing that a substantial amount of attention edges can be suppressed while retaining VideoQA performance.
- Script: [scripts/analysis/effective_pathway_analysis_*.sh](scripts/analysis)
- Implementation: [analysis/effective_pathway_analysis.py](analysis/effective_pathway_analysis.py)


### Logit Lens analysis

- Logit probing by projecting layerwise video token representations into language vocabulary space.
- Script: [scripts/analysis/logit_lens_analysis.sh](scripts/analysis/logit_lens_analysis.sh)
- Implementation: [analysis/logit_lens_analysis.py](analysis/logit_lens_analysis.py)
- After obtaining the json file, you can also run [analysis/visualize_logit_lens_vocab_frequency.py](analysis/visualize_logit_lens_vocab_frequency.py) to generate layer-wise vocabulary frequency plots (Fig. 4 in our paper):
- Add `--visualize_on_video` to generate per-frame visualizations (Fig. 5 in our paper). We use `task_id=3` (Action Sequence) in our paper.


### Attention map visualization

- Visualizes attention maps comparing baseline vs. attention knockout conditions (Fig. 6 in our paper).
- Script: [scripts/analysis/attention_visualization.sh](scripts/analysis/attention_visualization.sh)
- Implementation: [analysis/attention_visualization.py](analysis/attention_visualization.py)


## Training

If you want to reproduce our training process, please refer to [docs/TRAIN.md](docs/TRAIN.md).


## Acknowledgement

This project is built upon the following works:
- [dissecting_factual_predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions), [cross-modal-information-flow-in-MLLM](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM): Causal intervention analysis
- [PLLaVA](https://github.com/magic-research/PLLaVA): Base codebase and LLaVA-NeXT integration
- [InternVL](https://github.com/OpenGVLab/InternVL), [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3): Mini-InternVL and VideoLLaMA3 integration

We thank all authors who contributed to these foundational projects.


## Citation

If you find our paper useful in your research, please consider citing:

```bibtex
@inproceedings{kim2026map,
  author    = {Kim, Minji and Kim, Taekyung and Han, Bohyung},
  title     = {Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
}

@article{kim2025map,
  author    = {Kim, Minji and Kim, Taekyung and Han, Bohyung},
  title     = {Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs},
  journal   = {arXiv preprint arXiv:2510.13251},
  year      = {2025},
}
```

## Contact

If you have any questions, please create an issue or contact [minji@snu.ac.kr](mailto:minji@snu.ac.kr) and [taekyung.k@navercorp.com](mailto:taekyung.k@navercorp.com).


# Train

## Instruction Training Data

For training, we leveraged the video instruction tuning data from [VideoChat2-IT](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).
We only used the video part for training.
As some videos are missing, we performed annotation cleaning.
After downloading, set the paths in [tasks/train/instruction_data.py](../tasks/train/instruction_data.py).

#### 1. Download cleaned json annotation files from Hugging Face.
[![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VideoChat2_IT_clean-blue)](https://huggingface.co/datasets/byminji/VideoChat2-IT-clean)

Data cleaning was performed using [scripts/data_preprocess/clean_data_anno.py](../scripts/data_preprocess/clean_data_anno.py).
<details>
<summary>Data spec (total valid samples: 874,869)</summary>

| Video source |      Task      |      Dataset       |   Total |   Valid | Invalid |
|:------------:|:--------------:|:------------------:|--------:|--------:|--------:|
|    TextVR    |    caption     |       textvr       |  39,648 |  39,648 |       0 |
|   YouCook2   |    caption     |      youcook2      |   8,700 |   8,700 |       0 |
|   Kinetics   | classification |        k710        |  40,000 |  38,977 |   1,023 |
|     SSv2     | classification |        ssv2        |  40,000 |  40,000 |       0 |
|  InternVid   |  conversation  |     videochat2     |   9,584 |   9,584 |       0 |
| ActivityNet  |  conversation  |    videochatgpt    |  13,303 |  13,303 |       0 |
|   NExT-QA    |   reasoning    |      next_qa       |  34,132 |  34,132 |       0 |
|   CLEVRER    |   reasoning    |     clevrer_qa     |  40,000 |  40,000 |       0 |
|   CLEVRER    |   reasoning    |     clevrer_mc     |  40,000 |  40,000 |       0 |
|    EgoQA     |      vqa       |       ego_qa       |   7,813 |   7,797 |      16 |
|     TGIF     |      vqa       |   tgif_frame_qa    |  39,149 |  39,149 |       0 |
|     TGIF     |      vqa       | tgif_transition_qa |  52,696 |  52,696 |       0 |
|    WebVid    |    caption     |       webvid       | 400,000 | 399,740 |     260 |
|    WebVid    |    caption     |     videochat      |   6,889 |   6,889 |       0 |
|    WebVid    |  conversation  |     videochat1     |   4,300 |   4,300 |       0 |
|    WebVid    |      vqa       |     webvid_qa      | 100,000 |  99,954 |      46 |

</details>


<!-- > ![images](./assert/data.png) -->

#### 2. Download the raw videos from the following links.
The video directories can be found in [tasks/train/instruction_data.py](../tasks/train/instruction_data.py). You can also change them to your own saved paths.

- [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data), [EgoQA](https://ego4d-data.org/), [YouCook2](https://youcook2.eecs.umich.edu/): use below commands ([issues](https://github.com/OpenGVLab/Ask-Anything/issues/86#issuecomment-1882529070))
   ```bash
   # VideoChat
   wget https://huggingface.co/datasets/ynhe/videochat2_data/resolve/main/videochat2_conversation_videos.zip
   # YouCook2
   wget https://huggingface.co/datasets/ynhe/videochat2_data/resolve/main/youcook_split_videos.zip.partaa
   wget https://huggingface.co/datasets/ynhe/videochat2_data/resolve/main/youcook_split_videos.zip.partab
   cat youcook_split_videos.zip* >> youcook_split_videos.zip
   unzip youcook_split_videos.zip
   # EgoQA
   wget https://huggingface.co/datasets/ynhe/videochat2_data/resolve/main/egoqa_split_videos.zip
   ```
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data)
- [Kinetics-710](https://github.com/OpenGVLab/UniFormerV2/blob/main/DATASET.md), download Kinetics 400/600/700 [here](https://openxlab.org.cn/datasets?keywords=kinetics).
- [SthSthV2](https://developer.qualcomm.com/software/ai-datasets/something-something): Option candidates were generated from [UMT](https://github.com/OpenGVLab/unmasked_teacher) top-20 predictions.
- [NExTQA](https://github.com/doc-doc/NExT-QA)
- [CLEVRER](https://clevrer.csail.mit.edu/)
- [TextVR](https://github.com/callsys/textvr)
- [TGIF](https://github.com/YunseokJANG/tgif-qa)
- [WebVid](https://maxbain.com/webvid-dataset/)
  * The original site stopped hosting. Instead, videos must be crawled from the web.
  * Clone the WebVid-10M metadata repository from [TempoFunk/webvid-10M](https://huggingface.co/datasets/TempoFunk/webvid-10M) at `$metadata_repo`.
  * Run [scripts/data_preprocess/download_webvid.py](../scripts/data_preprocess/download_webvid.py) to download subsets of WebVid for training.
  * This script uses all available CPUs in parallel.
     ```bash
      # caption_videochat (6889 videos)
      python scripts/data_preprocess/download_webvid.py \
      --json workspace/annotations/VideoChat2-IT-clean/caption/videochat/train.json \
      --ann_dir $metadata_repo --save_dir $webvid_save_path/caption_videochat

      # conversation_videochat1 (4300 videos)
      python scripts/data_preprocess/download_webvid.py \
      --json workspace/annotations/VideoChat2-IT-clean/conversation/videochat1/train.json \
      --ann_dir $metadata_repo --save_dir $webvid_save_path/conversation_videochat1

      # caption_webvid (399820 videos)
      python scripts/data_preprocess/download_webvid.py \
      --json workspace/annotations/VideoChat2-IT-clean/caption/webvid/train.json \
      --ann_dir $metadata_repo --save_dir $webvid_save_path/caption_webvid

      # vqa_webvid_qa (99954 videos)
      python scripts/data_preprocess/download_webvid.py \
      --json workspace/annotations/VideoChat2-IT-clean/vqa/webvid_qa/train.json \
      --ann_dir $metadata_repo --save_dir $webvid_save_path/vqa_webvid_qa
     ```
  * After downloading, the total number of videos is around 5% of the full dataset.


## Training

Run scripts are provided under [scripts/train](../scripts/train).
Before running, set `repo_id` to the path of the base model and `output_dir` to your desired checkpoint directory.


| Model                     | Script                                                                                                                                                         | Initialized From                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| LLaVA-NeXT-7B-Video-FT    | [train_llava_next_7b_video_ft_8gpu.sh](../scripts/train/train_llava_next_7b_video_ft_8gpu.sh) | [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)           |
| LLaVA-NeXT-13B-Video-FT   | [train_llava_next_13b_video_ft_8gpu.sh](../scripts/train/train_llava_next_13b_video_ft_8gpu.sh) | [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)         |
| Mini-InternVL-4B-Video-FT | [train_mini_internvl_4b_video_ft_4gpu.sh](../scripts/train/train_mini_internvl_4b_video_ft_4gpu.sh) | [OpenGVLab/Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)  |

### Run

```bash
sh scripts/train/train_llava_next_7b_video_ft_8gpu.sh     # LLaVA-NeXT-7B on 8 GPUs (3epoch)
sh scripts/train/train_llava_next_13b_video_ft_8gpu.sh    # LLaVA-NeXT-13B on 8 GPUs (1epoch)
sh scripts/train/train_mini_internvl_4b_video_ft_4gpu.sh  # Mini-InternVL-4B on 4 GPUs (3epoch)
```

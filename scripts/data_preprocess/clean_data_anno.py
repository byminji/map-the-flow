"""
Script to clean up annotations by filtering missing videos & change video path formats
"""

import os
import shutil
import json
import argparse
from tqdm import tqdm
from decord import VideoReader, DECORDError


def parse_data(input_file, output_file, dataset_dir, check_loading=False):
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Create a new list for valid video annotations
    clean_data = []
    missing_videos = []
    unloadable_videos = []

    for item in tqdm(data):

        if not item['video']:
            unloadable_videos.append('none')
            continue

        # if video does not ends with .xxx, add .mp4
        if len(item['video'].split('.')) == 1:
            item['video'] = item['video'] + '.mp4'

        if 'clevrer' in input_file:
            # use filename only
            item['video'] = item['video'].split('/')[-1]
        elif 'videochatgpt' in input_file:
            # In case of activitynet
            if not os.path.exists(os.path.join(dataset_dir, item['video'])):
                item['video'] = item['video'].replace('mp4', 'mkv')

        video_path = os.path.join(dataset_dir, item['video'])

        # Exclude if file does not exist
        if os.path.exists(video_path):
            if check_loading:
                try:
                    # Attempt to load the video with decord
                    vr = VideoReader(video_path)
                    clean_data.append(item)
                    del vr
                except Exception as e:
                # except (DECORDError, RuntimeError) as e:
                    # If the file can't be loaded, add to unloadable_videos
                    unloadable_videos.append(item['video'])
                    print(e)
            else:
                clean_data.append(item)
        else:
            missing_videos.append(item['video'])

    # Save the updated data to another JSON file
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(clean_data, file, indent=4)

    # Print results
    print(f"Updated JSON file saved as {output_file}")
    print(f"Number of valid video annotations: {len(clean_data)}")
    print(f"Number of missing videos: {len(missing_videos)}")
    print(f"Number of unloadable videos: {len(unloadable_videos)}")
    # print("List of missing videos:")
    # for video in missing_videos:
    #     print(video)

    # Save missing videos to a sorted text file
    missing_videos = missing_videos + unloadable_videos
    if len(missing_videos) > 0:
        missing_videos_file = os.path.join(os.path.dirname(output_file), 'missing_videos.txt')
        with open(missing_videos_file, 'w') as file:
            for video in sorted(missing_videos):
                file.write(f"{video}\n")


def parse_data_k710(input_file, output_file, dataset_root):
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Create a dictionary to hold video IDs and corresponding filenames
    k400_video_dict = {}
    broken_folder = os.path.join(dataset_root, 'k400/broken_videos_train')
    os.makedirs(broken_folder, exist_ok=True)

    # Function to populate the hash table for a specific dataset
    def populate_video_dict(dataset_path):
        for file in tqdm(os.listdir(dataset_path)):
            video_id = os.path.splitext(file)[0]  # Get {id} from {id}_{timestamp1}_{timestamp2}.mp4
            prefix = video_id[:11]  # Get the video ID prefix
            video_path = os.path.join(dataset_path, file)

            # Exclude if file does not exist
            # try:
            #     # Attempt to load the video with decord
            #     vr = VideoReader(video_path)
            #     k400_video_dict[prefix] = video_path
            # except (DECORDError, RuntimeError) as e:
            #     # If the file can't be loaded, add to unloadable_videos
            #     print(e)
            #     broken_path = os.path.join(broken_folder, file)
            #     shutil.move(video_path, broken_path)
            #     continue

            k400_video_dict[prefix] = video_path

    # Populate the hash table for k400/train and k400/replacement
    populate_video_dict(os.path.join(dataset_root, 'k400/train'))
    # populate_video_dict(os.path.join(dataset_root, 'k400/replacement'))

    # Create a new list for valid video annotations
    clean_data = []
    missing_videos = []

    for item in tqdm(data):
        video = item['video']

        if not video:
            missing_videos.append(video)
            continue

        folders = video.split('/')

        if 'k400' in folders:
            # Before: k400/Bw6m04_IxTI.mp4
            # After: k400/train/Bw6m04_IxTI_xxxxxx_xxxxxx.mp4
            video_id = os.path.splitext(folders[-1])[0]
            prefix = video_id[:11]
            if prefix in k400_video_dict:
                item['video'] = os.path.relpath(k400_video_dict[prefix], dataset_root)
                clean_data.append(item)
            else:
                missing_videos.append(video)

            # rel_path = os.path.join('k400/train', folders[-1])
            # video_path = os.path.join(dataset_root, rel_path)
            # if os.path.exists(video_path):
            #     item['video'] = rel_path  # Keep the original relative path if it exists
            #     clean_data.append(item)
            # else:
            #     breakpoint()
            #     missing_videos.append(video)

        elif 'k600' in folders:
            # Before: p2:s3://k600/train_videos/eating_chips/xqnJh1oiEIU_000001_000011.mp4
            # After: k600/train/xqnJh1oiEIU_000001_000011.mp4
            rel_path = os.path.join('k600/train', folders[-1])
            video_path = os.path.join(dataset_root, rel_path)
            if os.path.exists(video_path):
                item['video'] = rel_path  # Keep the original relative path if it exists
                clean_data.append(item)
            else:
                missing_videos.append(video)

        elif 'k700' in folders:
            # Before: p2:s3://k700/train/swing_dancing/M4YR9XvLZ_I_000005_000015.mp4
            # After: k700/train/swing_dancing/M4YR9XvLZ_I_000005_000015.mp4
            rel_path = os.path.join('k700/train', folders[-2], folders[-1])
            video_path = os.path.join(dataset_root, rel_path)
            if os.path.exists(video_path):
                item['video'] = rel_path  # Keep the original relative path if it exists
                clean_data.append(item)
            else:
                missing_videos.append(video)
        else:
            missing_videos.append(video)

    # Save the updated data to another JSON file
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(clean_data, file, indent=4)

    # Print results
    print(f"Updated JSON file saved as {output_file}")
    print(f"Number of valid video annotations: {len(clean_data)}")
    print(f"Number of missing videos: {len(missing_videos)}")
    print("List of missing videos:")
    for video in missing_videos:
        print(video)

    # Save missing videos to a sorted text file
    missing_videos_file = os.path.join(os.path.dirname(output_file), 'missing_videos.txt')
    with open(missing_videos_file, 'w') as file:
        for video in sorted(missing_videos):
            file.write(f"{video}\n")


def main(args):
    anno_root_it = "workspace/annotations/VideoChat2-IT"
    output_root = "workspace/annotations/VideoChat2-IT-clean"

    data_root = "/your/path/to/datasets"

    available_corpus = dict(
        # caption
        caption_textvr=[
            f"{anno_root_it}/video/caption/textvr/train.json",
            f"{data_root}/TextVR/videos",
            "video"
        ],
        caption_youcook2=[
            f"{anno_root_it}/video/caption/youcook2/train.json",
            f"{data_root}/VideoChat2-IT/split_videos/youcook_split_videos",
            "video"
        ],
        caption_videochat=[
            f"{anno_root_it}/video/caption/videochat/train.json",
            f"{data_root}/WebVid-10M/caption_videochat",
            "video"
        ],
        caption_webvid=[
            f"{anno_root_it}/video/caption/webvid/train.json",
            f"{data_root}/WebVid-10M/caption_webvid",
            "video"
        ],

        # classification
        classification_k710=[
            f"{anno_root_it}/video/classification/k710/train.json",
            f"{data_root}",
            "video"
        ],
        classification_ssv2=[
            f"{anno_root_it}/video/classification/ssv2/train.json",
            f"{data_root}/ssv2/videos",
            "video"
        ],

        # conversation
        conversation_videochat1=[
            f"{anno_root_it}/video/conversation/videochat1/train.json",
            f"{data_root}/WebVid-10M/conversation_videochat1",
            "video"
        ],
        conversation_videochat2=[
            f"{anno_root_it}/video/conversation/videochat2/train.json",
            f"{data_root}/VideoChat2-IT/split_videos/videochat2_conversation_videos",
            "video"
        ],
        conversation_videochatgpt=[
            f"{anno_root_it}/video/conversation/videochatgpt/train.json",
            f"{data_root}/ActivityNet/source_activitynet_website/videos",
            "video"
        ],

        # reasoning
        reasoning_next_qa=[
            f"{anno_root_it}/video/reasoning/next_qa/train.json",
            f"{data_root}/NExT-QA/videos",
            "video"
        ],
        reasoning_clevrer_qa=[
            f"{anno_root_it}/video/reasoning/clevrer_qa/train.json",
            f"{data_root}/CLEVRER/train",
            "video"
        ],
        reasoning_clevrer_mc=[
            f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",
            f"{data_root}/CLEVRER/train",
            "video"
        ],

        # vqa
        vqa_ego_qa=[
            f"{anno_root_it}/video/vqa/ego_qa/train.json",
            f"{data_root}/VideoChat2-IT/split_videos/egoqa_split_videos",
            "video"
        ],
        vqa_tgif_frame_qa=[
            f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json",
            f"{data_root}/TGIF/gifs",
            "video"
        ],
        vqa_tgif_transition_qa=[
            f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json",
            f"{data_root}/TGIF/gifs",
            "video"
        ],
        vqa_webvid_qa=[
            f"{anno_root_it}/video/vqa/webvid_qa/train.json",
            f"{data_root}/WebVid-10M/vqa_webvid_qa",
            "video"
        ],
    )

    input_file, dataset_dir, data_type = available_corpus[args.ann]
    output_file = input_file.replace(anno_root_it, output_root)

    if args.ann == 'classification_k710':
        parse_data_k710(input_file, output_file, dataset_dir)
    else:
        parse_data(input_file, output_file, dataset_dir, args.check_loading)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video annotations based on specified dataset.')
    parser.add_argument('--ann', type=str, required=True,
                        help="Specify the dataset annotation type: e.g., 'classification_ssv2'.")
    parser.add_argument('--check_loading', action="store_true")
    args = parser.parse_args()
    main(args)

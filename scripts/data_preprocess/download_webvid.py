""" Script to download WebVid-10M corresponding to each subset"""

import os
import pandas as pd
import subprocess
import glob
import json
import argparse
import time

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_download_pairs(json_filename, ann_dir, base_dir):
    # Load the JSON data
    with open(json_filename, 'r') as file:
        data = json.load(file)

    # Extract unique video values and create a set
    video_set = {item['video'] for item in data}
    print(f"{json_filename}: video numbers: ", len(video_set))

    # Prepare a list for download pairs
    download_pairs = []
    # Set the partitions directory
    for split in ['train', 'val']:
        partitions_dir = os.path.join(ann_dir, f'data/{split}/partitions')
        print(f'partition_dir: {partitions_dir}')
        all_csv_files = sorted(glob.glob(os.path.join(partitions_dir, '*.csv')))

        for csv_file in tqdm(all_csv_files):
            df = pd.read_csv(csv_file)

            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                videoid = row['videoid']
                content_url = row['contentUrl']
                page_dir = row['page_dir']

                complete_path = f"{page_dir}/{videoid}.mp4"
                if complete_path in video_set:
                    output_file = os.path.join(base_dir, complete_path)
                    download_pairs.append((content_url, output_file))

    # Extract downloaded video paths from download_pairs
    download_videos = {pair[1] for pair in download_pairs}  # pair[1] contains the output file path

    # Check if all videos in video_set are included in download_videos
    missing_videos = {video for video in video_set if os.path.join(base_dir, video) not in download_videos}
    if not missing_videos:
        print("All videos in video_set are included in download_pairs.")
    else:
        print(f"Missing videos from download_pairs: {len(missing_videos)} videos")

    return download_pairs, video_set


def download_video(pair):
    content_url, output_file = pair
    print(f"Downloading {output_file}")

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Use wget to download the video
    subprocess.run(['wget', '-q', content_url, '-O', output_file])


def verify_downloads(download_pairs):
    print("Verifying downloads...")
    all_downloaded = True
    for _, output_file in tqdm(download_pairs):
        if not os.path.exists(output_file):
            print(f"Missing: {output_file}")
            all_downloaded = False
    return all_downloaded


def download_videos(json_filename, ann_dir, base_dir, check_downloads=False):
    # Create download pairs
    download_pairs, video_set = create_download_pairs(json_filename, ann_dir, base_dir)

    if check_downloads:
        # Verify if all videos are downloaded
        if verify_downloads(download_pairs):
            print("All videos have been successfully downloaded.")
        else:
            print("Some videos are missing.")
        return

    # Download videos using multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(download_video, pair): pair for pair in download_pairs}

        # Show progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # wait for each future to complete

    print("Process completed.")


def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV files.')
    parser.add_argument('--json', type=str, required=True, help='json path')
    parser.add_argument('--ann_dir', type=str, required=True, help='metadata path')
    parser.add_argument('--save_dir', type=str, required=True, help='dataset saving path')
    parser.add_argument('--check_download', action='store_true')

    args = parser.parse_args()

    """
    python scripts/data_preprocess/download_webvid.py \
    --json workspace/annotations/VideoChat2-IT-clean/video/caption/videochat/train.json \
    --ann_dir /your/path/to/metadata/webvid-10M --save_dir /your/path/to/datasets/WebVid-10M/caption_videochat
    
    python scripts/data_preprocess/download_webvid.py \
    --json workspace/annotations/VideoChat2-IT-clean/video/conversation/videochat1/train.json \
    --ann_dir /your/path/to/metadata/webvid-10M --save_dir /your/path/to/datasets/WebVid-10M/conversation_videochat1
    
    python scripts/data_preprocess/download_webvid.py \
    --json workspace/annotations/VideoChat2-IT-clean/video/caption/webvid/train.json \
    --ann_dir /your/path/to/metadata/webvid-10M --save_dir /your/path/to/datasets/WebVid-10M/caption_webvid
    
    python scripts/data_preprocess/download_webvid.py \
    --json workspace/annotations/VideoChat2-IT-clean/video/vqa/webvid_qa/train.json \
    --ann_dir /your/path/to/metadata/webvid-10M --save_dir /your/path/to/datasets/WebVid-10M/vqa_webvid_qa
    """

    start_time = time.time()

    download_videos(args.json, args.ann_dir, args.save_dir, args.check_download)

    # Calculate and print execution time
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"=============Downloaded videos in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()

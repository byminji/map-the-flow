from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
import json

from tasks.eval.config_dataset import DATASET_PATH


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=2)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    # Duplicate last indices if needed
    if len(frame_indices) < max_num_frames:
        last_index = frame_indices[-1] if frame_indices else 0
        frame_indices.extend([last_index] * (max_num_frames - len(frame_indices)))

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps


def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list


def insert_subtitles_into_frames(frames, frame_timestamps, subtitles,
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append(frame)
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        # print(frame_timestamp)
        interleaved_list.append(frame)

    return interleaved_list


class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 annotation_file="lvb_val.json",
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                 ):
        super().__init__()
        self.data_path = DATASET_PATH["longvideobench"]
        self.insert_text = insert_text

        with open(os.path.join(self.data_path, annotation_file)) as f:
            self.data_list = json.load(f)
        for idx, entry in enumerate(self.data_list):
            task_type = entry.pop("question_category")
            self.data_list[idx]['task_type'] = task_type
        self.max_num_frames = max_num_frames

    def __getitem__(self, index):
        di = self.data_list[index]
        task_type = di['task_type']

        inputs = []
        if self.max_num_frames == 0:
            ### No subtitles, no frames
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A") + di["correct_choice"]), "id": di["id"]}
        if self.max_num_frames == -1:
            ### All subtitles, no frames
            with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
                subtitles = json.load(f)
            inputs = insert_subtitles(subtitles)
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A") + di["correct_choice"]), "id": di["id"]}

        video_path = os.path.join(self.data_path, "videos", di["video_path"])
        frames, frame_timestamps = load_video(video_path, di["duration"],
                                              max_num_frames=self.max_num_frames)

        # with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
        #     subtitles = json.load(f)
        # inputs = []
        # if self.insert_text:
        #     # inputs = insert_subtitles_into_frames(frames, frame_timestamps, subtitles,
        #     #                                       di["starting_timestamp_for_subtitles"], di["duration"])
        #     raise NotImplementedError
        # else:
        #     inputs = frames

        # ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        # inputs += ["Question: " + di["question"]]
        # inputs += [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(di["candidates"])]
        # inputs += ["Answer with the option's letter from the given choices directly."]
        # ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####
        #
        # ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        # return {"inputs": inputs, "correct_choice": chr(ord("A") + di.get("correct_choice", -1)), "id": di["id"]}
        #
        question, answer = self.qa_template(di)

        return {
            "video_path": video_path,
            "video_pils": frames,
            'question': question,
            'answer': answer,
            'task_type': task_type,
        }

    def __len__(self):
        return len(self.data_list)

    def get_id(self, index):
        return self.data_list[index]["id"]

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer_idx = data['correct_choice']

        for idx, c in enumerate(data['candidates']):
            option = f"({chr(ord('A') + idx)}) {c}\n"
            question += option
            if answer_idx == idx:
                answer = c

        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

if __name__ == "__main__":
    db = LongVideoBenchDataset("../", "lvb_val.json")
    for i in range(10):
        print([ele for ele in db[i]["inputs"] if not isinstance(ele, str)])




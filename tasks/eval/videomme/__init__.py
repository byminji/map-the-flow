import os
import pyarrow.parquet as pq

from tasks.eval.eval_utils import EvalDataset

from tasks.eval.config_dataset import DATASET_PATH

# Tasks
# 'Action Reasoning', 'Action Recognition', 'Attribute Perception',
# 'Counting Problem', 'Information Synopsis', 'OCR Problems',
# 'Object Reasoning', 'Object Recognition',
# 'Spatial Perception', 'Spatial Reasoning',
# 'Temporal Perception', 'Temporal Reasoning'


class VideoMMEDataset(EvalDataset):
    def __init__(self, *args, **kwargs):
        # Remove 'add_options_template' from kwargs before passing to EvalDataset
        self.open_ended = kwargs.pop('open_ended', False)
        self.dataset_path = DATASET_PATH['videomme']

        super().__init__(*args, **kwargs)

        ##### Note that we omit subtitles integration

        self.data_list = self.load_data(data_root=self.dataset_path)

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

    def load_data(self, data_root: str):
        parquet_file = os.path.join(data_root, "videomme", "test-00000-of-00001.parquet")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        video_folder = os.path.join(data_root, "data")
        # subtitle_folder = os.path.join(data_root, "subtitles")

        data_list = []
        for record in df.itertuples():
            # video_id = record.videoID
            url = record.url
            video_id = url.split("https://www.youtube.com/watch?v=")[-1]
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                # temp_path = os.path.join(video_folder, f"{video_id}.{video_format}")
                # if os.path.exists(temp_path):
                #     video_path = temp_path
                #     break
                temp_path = os.path.join(video_folder, f"{video_id}.{video_format}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            assert os.path.exists(video_path), f"Cannot find the video file: {video_id}"

            meta_data = {
                # required fields for data loading
                "video_path": video_path,
                "start_time": None,
                "end_time": None,
                # required fields for evaluation
                "task_type": record.task_type,
                "ground_truth": record.answer,
                # custom fields for instruction generation and post processing
                "question": record.question,
                "options": list(record.options),
                "question_id": record.question_id,
                # other metadata
                "duration": record.duration,
                "domain": record.domain,
                "data_type": "video"
            }

            data_list.append(meta_data)

        return data_list

    def __getitem__(self, idx):
        data = self.data_list[idx]
        question = data['question'] #  When demonstrating the Germany modern Christma...
        options = data['options']   # [A. Apples., B. Candles., C. Berries., D. The ...]
        ground_truth = data['ground_truth'] # A or B or ...
        video_path = data['video_path']
        task_type = data['task_type']
        decord_method = self.decord_method[data['data_type']]

        question_full = f"Question: {question}\n"
        if not self.open_ended:
            question_full += "Options:\n"
            for option_idx, option in enumerate(options):
                question_full += f"{option}\n"

        question_without_options = data['question']
        true_option = ""
        false_options = []
        for option in options:
            if option.startswith(ground_truth):
                true_option = option
            else:
                false_options.append(option)

        try:  # might be problem with decord
            images_group = decord_method(video_path)
        except Exception as e:
            print(f'error decoding {video_path}')
            print(e)
            # task_type = 'error_reading_video'
            images_group = None

        return_dict = {
            'video_path': video_path,
            'video_pils': images_group,  # some might use the original pils and do their own transforms
            'question': question_full,
            'answer': true_option,
            'task_type': task_type,
            # added for ablation
            'question_without_options': question_without_options,
            'false_options': false_options,
        }

        return return_dict

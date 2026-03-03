import json
import os

from tasks.eval.eval_utils import EvalDataset
from tasks.eval.config_dataset import DATASET_PATH


class TOMATODataset(EvalDataset):
    def __init__(self, *args, **kwargs):
        self.open_ended = kwargs.pop('open_ended', False)

        super().__init__(*args, **kwargs)

        dataset_root = DATASET_PATH['tomato']
        self.video_folder = f'{dataset_root}/videos'
        self.ann_folder = f'{dataset_root}/jsons'

        self.data_list_info = {
            "Count": "count.json",
            "Direction": "direction.json",
            "Rotation": "rotation.json",
            "ShapeTrend": "shape&trend.json",
            "VelocityFrequency": "velocity&frequency.json",
            "VisualCues": "visual_cues.json"
        }

        self.data_list = []
        for k, v in self.data_list_info.items():
            with open(os.path.join(self.ann_folder, v), 'r') as f:
                json_data = json.load(f)
            for data in json_data.values():
                self.data_list.append({
                    'task_type': k,
                    'data_type': 'video',
                    'data': data
                })

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

    def __getitem__(self, idx):
        question, answer, false_options, bag_of_candidates = self.qa_template(self.data_list[idx]['data'])
        task_type = self.data_list[idx]['task_type']
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        video_path = os.path.join(self.video_folder, self.data_list[idx]['data']['demonstration_type'],
                                  '.'.join([self.data_list[idx]['data']['key'], 'mp4']))

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
            'question': question,
            'answer': answer,
            'task_type': task_type,
            # added for ablation
            # 'question_without_options': self.data_list[idx]['data']['question'],
            # 'false_options': false_options,
        }

        return return_dict

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        if not self.open_ended:
            question += "Options:\n"
        answer_idx = data['answer']
        answer = data['options'][answer_idx]
        if self.open_ended:
            return question, answer, None, None

        false_options = []
        bag_of_candidates = []
        for idx, c in enumerate(data['options']):
            option = f"({chr(ord('A') + idx)}) {c}\n"
            question += option
            bag_of_candidates.append(f"{chr(ord('A') + idx)}")
            if idx != answer_idx:
                false_options.append(option.rstrip())

        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer, false_options, bag_of_candidates
import json
import os
import re
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset


TASKS = {
            "Count": "count.json",
            "Direction": "direction.json",
            "Rotation": "rotation.json",
            "ShapeTrend": "shape&trend.json",
            "VelocityFrequency": "velocity&frequency.json",
            "VisualCues": "visual_cues.json"
        }


class TomatoDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"
    TASK_TYPES: List[str] = [task_type for task_type in TASKS]

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        for task_name, task_info in TASKS.items():
            json_file = os.path.join(data_root, 'jsons', task_info)
            video_folder = os.path.join(data_root, 'videos')

            with open(json_file, 'r') as f:
                task_data_dict = json.load(f)

            for data in task_data_dict.values():
                # answer = data["answer"]
                answer_idx = data["answer"]
                options = data["options"]

                option_letters = []
                option_letters_full = []
                for option_idx, option in enumerate(options):
                    option_letters.append(f"{chr(ord('A') + option_idx)}")
                    option_letters_full.append(f"({chr(ord('A') + option_idx)}) {option}\n")

                video_path = os.path.join(video_folder, data['demonstration_type'], '.'.join([data['key'], 'mp4']))
                print(video_path)

                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": video_path,
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                    "option_letters_full": option_letters_full
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        option_letters = meta_data["option_letters"]
        options = meta_data["options"]

        option_string = ""
        for option_idx, (letter, option) in enumerate(zip(option_letters, options)):
            option_string += f"({letter}) {option}\n"
        instruction = f"Question: {question}\nOptions:\n{option_string}Answer with the option\'s letter from the given choices directly and only give the best option."

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall(f'[\(,\ ]*[{option_letters[0]}-{option_letters[-1]}][\),\ ]*', response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                if opt.lower() in response.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = option_letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx

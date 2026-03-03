from collections import defaultdict
import os
import json
import re

from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)
from tasks.eval.config_dataset import DATASET_PATH


def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if not any([c in pred_option for c in 'abcdefgABCDEFG']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag


def save_results(result_list, save_path):

    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')


def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        result_list = all_results['result_list']
    else:
        result_list = None
    # json_data = load_json(save_path, 'all_results.json')['result_list']
    return result_list


class BaseDataset(EvalDataset):
    def __init__(self, data_list_info, data_dir, *args, **kwargs):
        self.open_ended = kwargs.pop('open_ended', False)

        super().__init__(*args, **kwargs)

        self.data_list_info = data_list_info
        self.data_dir = data_dir

        self.bag_of_words = defaultdict(set)
        self.bag_of_answer_object_words = {
            "Action Antonym": ['jacket', 'shoe', 'bag', 'headphone', 'hat/cap', 'glasses'],
            "Action Sequence": ['the floor', 'the sofa/couch', 'the laptop', 'the bag', 'the food', 'the book',
                                'the clothes', 'the paper/notebook', 'the phone/camera', 'the bed',
                                'the closet/cabinet', 'the towel', 'the dish', 'the refrigerator', 'the pillow',
                                'the door', 'the sandwich', 'the box', 'the cup/glass/bottle', 'the table',
                                'the blanket']
        }
        self.bag_of_question_action_words = {
            "Action Antonym": ['the action'],
            "Action Localization": ['what moment', 'When', 'when', 'which part',
                                    'occur', 'observe', 'happens', 'take place',
                                    'the action'],
            "Action Sequence": ['do first'],
            "Moving Direction": ['direction', 'move'],
            "Object Count": ["How many", "moving", "when", "begins", "ends"]
        }
        self.bag_of_question_object_words = {
            "Action Sequence": ['the person'],
            "Moving Direction": ['the cyan sphere', 'the cyan cylinder', 'the cyan cube',
                                 'the green sphere', 'the green cylinder', 'the green cube',
                                 'the blue sphere', 'the blue cylinder', 'the blue cube',
                                 'the gray sphere', 'the gray cylinder', 'the gray cube',
                                 'the purple sphere', 'the purple cylinder', 'the purple cube',
                                 'the yellow sphere', 'the yellow cylinder', 'the yellow cube',
                                 'the brown sphere', 'the brown cylinder', 'the brown cube',
                                 'the red sphere', 'the red cylinder', 'the red cube'],
            "Object Count": ["metal objects", "rubber objects", "cylinders", "objects"]
        }

        self.data_list = []
        for k, v in self.data_list_info.items():
            with open(os.path.join(self.data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                # Open-ended
                answer_prompt = None
                candidates = []
                if self.open_ended:
                    if k == "Action Antonym":
                        object_keyword = ""
                        for object in self.bag_of_answer_object_words[k]:
                            if object in data['answer']:
                                object_keyword = object
                                break
                        if object_keyword is not "":
                            answer_prompt = f"\nThe action being performed in the video regarding the {object_keyword} is to"
                        else:
                            answer_prompt = f"\nThe action being performed in the video is to"
                        data['answer'] = data['answer'].lower()
                        for candidate in data['candidates']:
                            candidates.append(candidate.lower())

                    elif k == "Action Sequence":
                        answer_prompt = "\nThe action the person is doing first is to"
                        data['answer'] = data['answer'].lower()
                        for candidate in data['candidates']:
                            candidates.append(candidate.lower())

                    elif k == "Action Localization":
                        match = re.search(r"action '(.*?)'", data['question'])
                        action = match.group(1)
                        answer_prompt = f"\nThe action '{action}' occurs at the"
                        for answer in ['Throughout', 'beginning', 'middle', 'end']:
                            if answer in data['answer']:
                                break
                        data['answer'] = answer.lower()
                        candidates = ['throughout', 'beginning', 'middle', 'end']

                    elif k == "Moving Direction":
                        object_keyword = "the object"
                        for object in self.bag_of_question_object_words[k]:
                            if object in data['question']:
                                object_keyword = object
                                break
                        object_keyword = object_keyword[:1].upper() + object_keyword[1:]
                        answer_prompt = f"\n{object_keyword} moves to the"
                        if 'right' in data['answer']:
                            data['answer'] = 'right'
                        else:
                            data['answer'] = 'left'
                        candidates = ['left', 'right']

                    elif k == "Object Count":
                        answer_prompt = "\nThe number of moving objects is"
                        number_to_word = {
                            '0': 'zero',
                            '1': 'one',
                            '2': 'two',
                            '3': 'three',
                            '4': 'four',
                            '5': 'five',
                            '6': 'six',
                            '7': 'seven',
                            '8': 'eight',
                            '9': 'nine'
                        }
                        data['answer'] = number_to_word[data['answer']]
                        for candidate in data['candidates']:
                            candidates.append(number_to_word[candidate])

                    elif k == "Scene Transition":
                        data["question"] = "Where does the scene in the video change from one place to another place?"
                        answer_prompt = "\nThe scene in the video changes from the"
                        data['answer'] = data["answer"].lower()
                        if data['answer'].startswith("from the"):
                            data['answer'] = data['answer'].replace("from the ", "")
                        elif data['answer'].startswith("from a"):
                            data['answer'] = data['answer'].replace("from a ", "")
                        elif data['answer'].startswith("from"):
                            data['answer'] = data['answer'].replace("from ", "")

                        for candidate in data['candidates']:
                            candidate = candidate.lower()
                            if candidate.startswith("from the"):
                                candidate = candidate.replace("from the ", "")
                            elif candidate.startswith("from a"):
                                candidate = candidate.replace("from a ", "")
                            elif candidate.startswith("from"):
                                candidate = candidate.replace("from ", "")
                            candidates.append(candidate)

                data['answer_prompt'] = answer_prompt
                data['bag_of_candidates'] = candidates

                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
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
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])

        try:  # might be problem with decord
            images_group, frame_indices = decord_method(video_path, bound, return_index=True)
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
            'question_without_options': self.data_list[idx]['data']['question'],
            'false_options': false_options,
            'answer_prompt': self.data_list[idx]['data']['answer_prompt'],
            'candidates': bag_of_candidates if not self.open_ended else self.data_list[idx]['data']['bag_of_candidates']
        }
        return return_dict

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        if not self.open_ended:
            question += "Options:\n"
        answer = data['answer']
        if self.open_ended:
            return question, answer, None, None

        false_options = []
        bag_of_candidates = []
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            option = f"({chr(ord('A') + idx)}) {c}\n"
            question += option
            bag_of_candidates.append(f"{chr(ord('A') + idx)}")
            if c == answer:
                answer_idx = idx
            else:
                false_options.append(option.rstrip())

        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer, false_options, bag_of_candidates


class MVBenchDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        print("Initializing MVBenchDataset ...")
        dataset_root = DATASET_PATH['mvbench']
        data_list_info = {
            "Action Sequence": ("action_sequence.json", f"{dataset_root}/video/star/Charades_v1_480/", "video", True), # has start & end
            "Action Prediction": ("action_prediction.json", f"{dataset_root}/video/star/Charades_v1_480/", "video", True), # has start & end
            "Action Antonym": ("action_antonym.json", f"{dataset_root}/video/ssv2_video/", "video", False),
            "Fine-grained Action": ("fine_grained_action.json", f"{dataset_root}/video/Moments_in_Time_Raw/videos/", "video", False),
            "Unexpected Action": ("unexpected_action.json", f"{dataset_root}/video/FunQA_test/test/", "video", False),
            "Object Existence": ("object_existence.json", f"{dataset_root}/video/clevrer/video_validation/", "video", False),
            "Object Interaction": ("object_interaction.json", f"{dataset_root}/video/star/Charades_v1_480/", "video", True), # has start & end
            "Object Shuffle": ("object_shuffle.json", f"{dataset_root}/video/perception/videos/", "video", False),
            "Moving Direction": ("moving_direction.json", f"{dataset_root}/video/clevrer/video_validation/", "video", False),
            "Action Localization": ("action_localization.json", f"{dataset_root}/video/sta/sta_video/", "video", True),  # has start & end
            "Scene Transition": ("scene_transition.json", f"{dataset_root}/video/scene_qa/video/", "video", False),
            "Action Count": ("action_count.json", f"{dataset_root}/video/perception/videos/", "video", False),
            "Moving Count": ("moving_count.json", f"{dataset_root}/video/clevrer/video_validation/", "video", False),
            "Moving Attribute": ("moving_attribute.json", f"{dataset_root}/video/clevrer/video_validation/", "video", False),
            "State Change": ("state_change.json", f"{dataset_root}/video/perception/videos/", "video", False),
            "Fine-grained Pose": ("fine_grained_pose.json", f"{dataset_root}/video/nturgbd/", "video", False),
            "Character Order": ("character_order.json", f"{dataset_root}/video/perception/videos/", "video", False),
            "Egocentric Navigation": ("egocentric_navigation.json", f"{dataset_root}/video/vlnqa/", "video", False),
            "Episodic Reasoning": ("episodic_reasoning.json", f"{dataset_root}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
            "Counterfactual Inference": ("counterfactual_inference.json", f"{dataset_root}/video/clevrer/video_validation/", "video", False),
        }
        data_dir = f"{dataset_root}/json"
        super().__init__(data_list_info, data_dir, *args, **kwargs)


class TVBenchDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        print("Initializing TVBenchDataset ...")
        dataset_root = DATASET_PATH['tvbench']
        data_list_info = {
            "Action Count": ("action_count.json", f"{dataset_root}/video/action_count", "video", False),
            "Object Count": ("object_count.json", f"{dataset_root}/video/object_count", "video", False),
            "Action Sequence": ("action_sequence.json", f"{dataset_root}/video/action_sequence", "video", True),  # has start & end
            "Object Shuffle": ("object_shuffle.json", f"{dataset_root}/video/object_shuffle", "video", False),
            "Scene Transition": ("scene_transition.json", f"{dataset_root}/video/scene_transition", "video", False),
            "Action Localization": ("action_localization.json", f"{dataset_root}/video/action_localization", "video", True),  # has start & end
            "Action Antonym": ("action_antonym.json", f"{dataset_root}/video/action_antonym", "video", False),
            "Unexpected Action": ("unexpected_action.json", f"{dataset_root}/video/unexpected_action", "video", False),
            "Egocentric Sequence": ("egocentric_sequence.json", f"{dataset_root}/video/egocentric_sequence", "video", False),
            "Moving Direction": ("moving_direction.json", f"{dataset_root}/video/moving_direction", "video", False),
        }
        data_dir = f"{dataset_root}/json"

        super().__init__(data_list_info, data_dir, *args, **kwargs)

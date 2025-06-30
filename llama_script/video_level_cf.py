# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import os
# local_rank = int(os.environ["LOCAL_RANK"])
local_rank = int(os.environ["SLURM_LOCALID"])
from typing import List, Optional

import fire

from llama import Dialog, Llama

import pandas as pd
from tqdm import tqdm
import json
import pickle
import os.path
import random

def stringtolist(description):
    outputs = {}

    if "[CF 1]:" in description and not '**' in description:
        outputs["CF 1"] = description.split("[CF 1]:")[1].split("[CF 2]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 1]:**" in description:
        outputs["CF 1"] = description.split("**[CF 1]:**")[1].split("**[CF 2]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 1]**" in description:
        outputs["CF 1"] = description.split("**[CF 1]**")[1].split("**[CF 2]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 2]:" in description and not '**' in description:
        outputs["CF 2"] = description.split("[CF 2]:")[1].split("[CF 3]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 2]:**" in description:
        outputs["CF 2"] = description.split("**[CF 2]:**")[1].split("**[CF 3]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 2]**" in description:
        outputs["CF 2"] = description.split("**[CF 2]**")[1].split("**[CF 3]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 3]:" in description and not '**' in description:
        outputs["CF 3"] = description.split("[CF 3]:")[1].split("[CF 4]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 3]:**" in description:
        outputs["CF 3"] = description.split("**[CF 3]:**")[1].split("**[CF 4]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 3]**" in description:
        outputs["CF 3"] = description.split("**[CF 3]**")[1].split("**[CF 4]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 4]:" in description and not '**' in description:
        outputs["CF 4"] = description.split("[CF 4]:")[1].split("[CF 5]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 4]:**" in description:
        outputs["CF 4"] = description.split("**[CF 4]:**")[1].split("**[CF 5]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 4]**" in description:
        outputs["CF 4"] = description.split("**[CF 4]**")[1].split("**[CF 5]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 5]:" in description and not '**' in description:
        outputs["CF 5"] = description.split("[CF 5]:")[1].split("[CF 6]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 5]:**" in description:
        outputs["CF 5"] = description.split("**[CF 5]:**")[1].split("**[CF 6]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 5]**" in description:
        outputs["CF 5"] = description.split("**[CF 5]**")[1].split("**[CF 6]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 6]:" in description and not '**' in description:
        outputs["CF 6"] = description.split("[CF 6]:")[1].split("[CF 7]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 6]:**" in description:
        outputs["CF 6"] = description.split("**[CF 6]:**")[1].split("**[CF 7]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 6]**" in description:
        outputs["CF 6"] = description.split("**[CF 6]**")[1].split("**[CF 7]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 7]:" in description and not '**' in description:
        outputs["CF 7"] = description.split("[CF 7]:")[1].split("[CF 8]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 7]:**" in description:
        outputs["CF 7"] = description.split("**[CF 7]:**")[1].split("**[CF 8]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 7]**" in description:
        outputs["CF 7"] = description.split("**[CF 7]**")[1].split("**[CF 8]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 8]:" in description and not '**' in description:
        outputs["CF 8"] = description.split("[CF 8]:")[1].split("[CF 9]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 8]:**" in description:
        outputs["CF 8"] = description.split("**[CF 8]:**")[1].split("**[CF 9]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 8]**" in description:
        outputs["CF 8"] = description.split("**[CF 8]**")[1].split("**[CF 9]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 9]:" in description and not '**' in description:
        outputs["CF 9"] = description.split("[CF 9]:")[1].split("[CF 10]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 9]:**" in description:
        outputs["CF 9"] = description.split("**[CF 9]:**")[1].split("**[CF 10]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 9]**" in description:
        outputs["CF 9"] = description.split("**[CF 9]**")[1].split("**[CF 10]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[CF 10]:" in description and not '**' in description:
        outputs["CF 10"] = description.split("[CF 10]:")[1].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 10]:**" in description:
        outputs["CF 10"] = description.split("**[CF 10]:**")[1].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 10]**" in description:
        outputs["CF 10"] = description.split("**[CF 10]**")[1].strip('\n').strip(' ').strip('\n').strip('- ')
    return outputs

def evenly_sample(lst, num_samples):
    if num_samples <= 0:
        return []
    step = max(1, len(lst) // num_samples)
    return lst[::step][:num_samples]

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    split: int = 0,
    device: int = 0
):

    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    # torch.cuda.set_device(device)
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    metadata = pd.read_csv('../egosummary_full.csv', sep='\t', on_bad_lines = "skip")
    with open('../summary_clips_hierarchy_full.json') as f:
        hier_metadata = json.load(f)
    num_summary = len(metadata)
    dataset_summary = []
    dataset_narration = []
    dataset_video_id = []
    video_narration = []
    video_id = metadata.iloc[0][0]
    pass_id = metadata.iloc[0][2]
    total_narrations = 0
    for s in tqdm(range(0, num_summary)):
        # if video_id != metadata.iloc[f][0] or pass_id != metadata.iloc[f][2]:
        video_id = metadata.iloc[s][0]
        summary = metadata.iloc[s][7]
        # if summary in hier_metadata:
        clip_narration_list = hier_metadata[summary]['clip_text']
        total_narrations += len(clip_narration_list)
        clip_narration = ''
        if len(clip_narration_list) != 0:
            if len(clip_narration_list) >= 60:
                clip_narration_list = evenly_sample(clip_narration_list, 30)
            clip_narration = clip_narration_list[0].strip(' ').strip('\n').strip(' ').strip('\n')
            for n in range(1, len(clip_narration_list)):
                update_narration = clip_narration_list[n].strip(' ').strip('\n').strip(' ').strip('\n')
                if n != len(clip_narration_list)-1:
                    clip_narration = os.path.join(clip_narration,'. ',update_narration,)
                else:
                    clip_narration = os.path.join(clip_narration,'. ',update_narration,'.')

        dataset_narration.append(clip_narration)
        dataset_summary.append(summary)
    print(f"\n=============  avg num narrations:{total_narrations/num_summary}  =====================\n")
    dialogs: List[Dialog] = []

    split_total_num = num_summary//16
    split_start = split*split_total_num
    split_end = (split+1)*split_total_num
    if split == 15:
        split_end = num_summary
    print(0)
    if os.path.exists(f'summary_counterfactual/order_16_{split}_progress.json'):
        with open(f'summary_counterfactual/order_16_{split}_progress.json', 'r') as f:
            order_temp = json.load(f)
    else:
        with open(f'summary_counterfactual/order_16_{split}.json', 'r') as f:
            order_temp = json.load(f)
    if os.path.exists(f'summary_counterfactual/key_16_{split}_progress.json'):
        with open(f'summary_counterfactual/key_16_{split}_progress.json', 'r') as f:
            key_temp = json.load(f)
    else:
        with open(f'summary_counterfactual/key_16_{split}.json', 'r') as f:
            key_temp = json.load(f)

    print(1)
    new_summary_dataset = []
    for i in tqdm(range(split_start, split_end)):
        s_temp = metadata.iloc[i][7]
        new_summary_dataset.append(s_temp)
        # if os.path.exists(f'order_error_16_{split}_progress.json'):
        if s_temp in order_temp.keys() and s_temp in key_temp.keys():
            # or s_temp in order_error_temp.keys() or s_temp in key_error_temp
            continue
        if dataset_narration[i] != '':
            dialogs.append(
                [
                    {"role": "system", "content": """\
                    Given a sequence of narrations describing a long video, and a video-level summary,\
                    create 10 distinct counterfactual summary [CF] with one to two sentences by perturbing the order of narrations.\
                    Follow this exact format to output:
                    [CF 1]: ...\
                    [CF 2]: ...\
                    [CF 3]: ...\
                    [CF 4]: ...\
                    [CF 5]: ...\
                    [CF 6]: ...\
                    [CF 7]: ...\
                    [CF 8]: ...\
                    [CF 9]: ...\
                    [CF 10]: ...
                    """},
                    {"role": "user", "content": f"""Here is the video-level summary: "{dataset_summary[i]}" and here is the sequence of narrations: "{dataset_narration[i]}. If the summary or narrations or contain harmful information, just modify it to a safe scenario."
                    """},
                ]
            )
            dialogs.append(
                [
                    {"role": "system", "content": """\
                    Given a sequence of narrations describing a long video, and a video-level summary,\
                    create 10 distinct counterfactual summary [CF] with one to two sentences by taking out some critical narrations.\
                    Follow this exact format to output:
                    [CF 1]: ...\
                    [CF 2]: ...\
                    [CF 3]: ...\
                    [CF 4]: ...\
                    [CF 5]: ...\
                    [CF 6]: ...\
                    [CF 7]: ...\
                    [CF 8]: ...\
                    [CF 9]: ...\
                    [CF 10]: ...
                    Note that no need to output how you perturbed.
                    """},
                    {"role": "user", "content": f"""Here is the video-level summary: "{dataset_summary[i]}" and here is the sequence of narrations: "{dataset_narration[i]}. If the summary or narrations or contain harmful information, just modify it to a safe scenario."
                    """},
                ]
            )
        else:
            dialogs.append(
                [
                    {"role": "system", "content": """\
                    Given a video-level summary,\
                    create 10 distinct counterfactual summary [CF] with one to two sentences by perturbing the order of activity stpes that the summary may need.\
                    Follow this exact format to output:
                    [CF 1]: ...\
                    [CF 2]: ...\
                    [CF 3]: ...\
                    [CF 4]: ...\
                    [CF 5]: ...\
                    [CF 6]: ...\
                    [CF 7]: ...\
                    [CF 8]: ...\
                    [CF 9]: ...\
                    [CF 10]: ...
                    """},
                    {"role": "user", "content": f"""Here is the video-level summary: "{dataset_summary[i]}" and here is the sequence of narrations: "{dataset_narration[i]}. If the summary or narrations or contain harmful information, just modify it to a safe scenario."
                    """},
                ]
            )
            dialogs.append(
                [
                    {"role": "system", "content": """\
                    Given a video-level summary,\
                    create 10 distinct counterfactual summary [CF] with one to two sentences by taking out some critical activity steps that the summary may need.\
                    Follow this exact format to output:
                    [CF 1]: ...\
                    [CF 2]: ...\
                    [CF 3]: ...\
                    [CF 4]: ...\
                    [CF 5]: ...\
                    [CF 6]: ...\
                    [CF 7]: ...\
                    [CF 8]: ...\
                    [CF 9]: ...\
                    [CF 10]: ...
                    Note that no need to output how you perturbed.
                    """},
                    {"role": "user", "content": f"""Here is the video-level summary: "{dataset_summary[i]}" and here is the sequence of narrations: "{dataset_narration[i]}. If the summary or narrations or contain harmful information, just modify it to a safe scenario."
                    """},
                ]
            )

    order_save_dict = order_temp
    key_save_dict = key_temp
    order_error_dict = {}
    key_error_dict = {}

    for i in tqdm(range(0, len(new_summary_dataset))):
        print(f' {i} / {len(new_summary_dataset)}')
        results = generator.chat_completion(
            dialogs[i:i+2],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        out_of_order_results = results[0]['generation']['content']
        missing_key_results = results[1]['generation']['content']
        # ----------------------------
        parsed_order_results = stringtolist(out_of_order_results)
        parsed_key_results = stringtolist(missing_key_results)
        # ----------------------------
        parsed_key = ['CF 1', 'CF 2', 'CF 3', 'CF 4', 'CF 5', 
                    'CF 6', 'CF 7', 'CF 8', 'CF 9', 'CF 10']
        # ----------------------------
        if set(parsed_key) != set(list(parsed_order_results.keys())):
            print(parsed_order_results.keys())
            order_error_dict[new_summary_dataset[i]] = ''
            # print(new_summary_dataset[i])
            print(out_of_order_results)
        else:
            order_save_dict[new_summary_dataset[i]] = {'CF 1': parsed_order_results['CF 1'],
                                        'CF 2': parsed_order_results['CF 2'],
                                        'CF 3': parsed_order_results['CF 3'],
                                        'CF 4': parsed_order_results['CF 4'],
                                        'CF 5': parsed_order_results['CF 5'],
                                        'CF 6': parsed_order_results['CF 6'],
                                        'CF 7': parsed_order_results['CF 7'],
                                        'CF 8': parsed_order_results['CF 8'],
                                        'CF 9': parsed_order_results['CF 9'],
                                        'CF 10': parsed_order_results['CF 10']
                                        }
            with open(f'summary_counterfactual/order_16_{split}_progress.json', 'w') as f:
                json.dump(order_save_dict, f)
        # ----------------------------       
        print(f"\n=============================\n")
        if set(parsed_key) != set(list(parsed_key_results.keys())):
            key_error_dict[new_summary_dataset[i]] = ''
            # print(new_summary_dataset[i])
            print(missing_key_results)
        else:
            key_save_dict[new_summary_dataset[i]] = {'CF 1': parsed_key_results['CF 1'],
                                        'CF 2': parsed_key_results['CF 2'],
                                        'CF 3': parsed_key_results['CF 3'],
                                        'CF 4': parsed_key_results['CF 4'],
                                        'CF 5': parsed_key_results['CF 5'],
                                        'CF 6': parsed_key_results['CF 6'],
                                        'CF 7': parsed_key_results['CF 7'],
                                        'CF 8': parsed_key_results['CF 8'],
                                        'CF 9': parsed_key_results['CF 9'],
                                        'CF 10': parsed_key_results['CF 10']
                                        }
            with open(f'summary_counterfactual/key_16_{split}_progress.json', 'w') as f:
                json.dump(key_save_dict, f)

            
        print(f"\n=============  order  ================\n")
        print(f"\n=============  num_good_video:{len(order_save_dict)}  =====================\n")
        print(f"\n=============  num_error_video:{len(order_error_dict)}  =====================\n")
        print(f"\n=============  key  ================\n")
        print(f"\n=============  num_good_video:{len(key_save_dict)}  =====================\n")
        print(f"\n=============  num_error_video:{len(key_error_dict)}  =====================\n")


    # metadata.to_csv('states.csv', index=True)
    with open(f'summary_counterfactual/order_16_{split}.json', 'w') as f:
        json.dump(order_save_dict, f)
    with open(f'summary_counterfactual/key_16_{split}.json', 'w') as f:
        json.dump(key_save_dict, f)




if __name__ == "__main__":
    # local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    fire.Fire(main)

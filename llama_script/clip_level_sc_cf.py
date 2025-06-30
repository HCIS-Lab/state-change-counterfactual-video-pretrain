# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import os
local_rank = int(os.environ["LOCAL_RANK"])

from typing import List, Optional

import fire

from llama import Dialog, Llama

import pandas as pd
from tqdm import tqdm
import json
import pickle
import os.path


def stringtolist(description):
    outputs = {}

    if "[Before]:" in description and not '**' in description:
        outputs["before"] = description.split("[Before]:")[1].split("[After]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[Before]:**" in description:
        outputs["before"] = description.split("**[Before]:**")[1].split("**[After]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[Before]**" in description:
        outputs["before"] = description.split("**[Before]**")[1].split("**[After]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

    if "[After]:" in description and not '**' in description:
        outputs["after"] = description.split("[After]:")[1].split("[CF 1]:")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[After]:**" in description:
        outputs["after"] = description.split("**[After]:**")[1].split("**[CF 1]:**")[0].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[After]**" in description:
        outputs["after"] = description.split("**[After]**")[1].split("**[CF 1]**")[0].strip('\n').strip(' ').strip('\n').strip('- ')

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
        outputs["CF 3"] = description.split("[CF 3]:")[1].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 3]:**" in description:
        outputs["CF 3"] = description.split("**[CF 3]:**")[1].strip('\n').strip(' ').strip('\n').strip('- ')
    elif "**[CF 3]**" in description:
        outputs["CF 3"] = description.split("**[CF 3]**")[1].strip('\n').strip(' ').strip('\n').strip('- ')

    return outputs


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

    if not os.path.isfile('stored_narration_set.pkl'):
        # metadata = pd.read_csv('/work/u8526971/ego4d/egoclip.csv', sep='\t', on_bad_lines = "skip")
        # metadata = pd.read_csv('/media/user/volume_0/ego4d/egoclip.csv', sep='\t', on_bad_lines = "skip")
        metadata = pd.read_csv('/media/user/data2/ego4d_data/v2/egoclip.csv', sep='\t', on_bad_lines = "skip")
        num_clips = len(metadata)
        narration = []
        for f in tqdm(range(0, num_clips)):
            narration.append(metadata.iloc[f][7].strip(' ').strip('\n').strip(' ').strip('\n'))
        narration = list(set(narration))
        narration.sort()
        with open("stored_narration_set.pkl", "wb") as fp:   #Pickling
            pickle.dump(narration, fp)
    else:
        with open("stored_narration_set.pkl", "rb") as fp:   # Unpickling
            narration = pickle.load(fp)


    dialogs: List[Dialog] = []
    for i in tqdm(range(len(narration))):
        dialogs.append(
            [
                {"role": "system", "content": """\
                Given a narration describing an action captured by camera wearer #C, the action may be performed by C or other participants, such as H, O, X, or Y.\
                Firstly, generate one [Before] describing the scene before action performed.\
                Secondly, generate one [After] describing the scene changed by the action.\
                Thirdly, create 3 distinct counterfactual descriptions (CF): [CF 1], [CF 2], and [CF 3]. The counterfactual could be incomplete execution of action or complete an action the wrong way.\
                Do not reuse the same verb in the narration. Note that the narration does not contain any harmful, illegal, or sexual activity"""},
                {"role": "user", "content": f"""Here's an example:\
                The narration: "#C C picks a bag of clothes from the floor."\
                \
                [Before]:\
                - The floor is cluttered with clothes.\
                [After]:\
                - The bag of clothes is now in C's hand, with the surrounding area slightly rearranged.\
                [CF 1]:\
                - Clothes remain scattered on the floor.\
                [CF 2]:\
                - A small pile of clothes sits amidst remaining clutter.\
                [CF 3]:\
                - The room is now even messier than before.\
                \
                Now, generate [Before], [After], [CF 1], [CF 2], and [CF 3] for the narration: "{narration[i]}" with the exact same format from the example above. 
                """},

            ]
        )

    num_clips = len(dialogs)
    split_total_num = num_clips//16
    split_start = split*split_total_num
    split_end = (split+1)*split_total_num
    if split == 15:
        split_end = num_clips
    save_dict = {}
    save_error_dict = {}

    for i in tqdm(range(split_start, split_end, max_batch_size)):

        results = generator.chat_completion(
            dialogs[i:i+max_batch_size],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for batch_idx, result in enumerate(results):
            states = stringtolist(result['generation']['content'])
            if not 'before' in states or not 'after' in states or not 'CF 1' in states or not 'CF 2' in states or not 'CF 3' in states:
                save_error_dict[narration[i+batch_idx]] = ''
                print(narration[i+batch_idx])
                print(states)
                print(result['generation']['content'])
            else:
                save_dict[narration[i+batch_idx]] = {'Before': states['before'],
                                        'After': states['after'],
                                        'CF 1': states['CF 1'],
                                        'CF 2': states['CF 2'],
                                        'CF 3': states['CF 3']}

        print(f"\n=============  num_good_clips:{len(save_dict)}  =====================\n")
        print(f"\n=============  num_error_clips:{len(save_error_dict)}  =====================\n")
        with open(f'states_error_{split}.json', 'w') as f:
            json.dump(save_error_dict, f)
    # metadata.to_csv('states.csv', index=True)
    with open(f'states_{split}.json', 'w') as f:
        json.dump(save_dict, f)
    with open(f'states_error_{split}.json', 'w') as f:
        json.dump(save_error_dict, f)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    fire.Fire(main)

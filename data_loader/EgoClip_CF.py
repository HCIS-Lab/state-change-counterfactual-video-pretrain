# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import pandas as pd

from base.base_dataset import TextVideoDataset
try:
    from data_loader.transforms import init_transform_dict, init_video_transform_dict
except:
    from transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms
import random
from tqdm import tqdm
import numpy as np

class EgoClip_CF(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'egoclip.csv',

        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t', on_bad_lines='skip')
            self.frame_sample = 'rand'

            self.state_metadata = 


            append_summary_baseline = False #TODO: Move to config
            if append_summary_baseline:
                summary_target_splits_fp = 'egosummary_full.csv'
                self.summary_metadata = pd.read_csv(os.path.join(self.meta_dir, summary_target_splits_fp), sep='\t', on_bad_lines='skip')
                print('Assigning summary duration to one of the clips...')
                for summary_idx in range(len(self.summary_metadata)):
                    self.summary_metadata.loc[summary_idx, "clip_start"] = random.uniform(0.0, self.summary_metadata.iloc[summary_idx]['clip_start']-4.0)
                    self.summary_metadata.loc[summary_idx, "clip_end"] = self.summary_metadata.iloc[summary_idx]['clip_start'] + random.uniform(1.5, 3.0)
                    # self.summary_metadata.iloc[summary_idx]['clip_start'] = random.uniform(0.0, self.summary_metadata.iloc[summary_idx]['clip_start']-4.0)
                    # self.summary_metadata.iloc[summary_idx]['clip_end'] = self.summary_metadata.iloc[summary_idx]['clip_start'] + random.uniform(1.5, 3.0) #typical clip duration
                self.metadata = pd.concat([self.metadata, self.summary_metadata], ignore_index=True)

            if self.neg_param:
                self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
                self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)
                self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + self.metadata['chunk_id']


    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_dir, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_dir, video_uid, str(chunk_end_id) + ".mp4")

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec

    def _get_video_frames(self, video_fp, video_sec, bound_sec):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, idxs = self.video_reader(video_fp[0], video_fp[1], self.video_params['num_frames'], self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], bound_sec=bound_sec)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample['clip_text'], noun_vec, verb_vec

    def _get_state_features(self, video_filename):
        # example filename: 0e3ee603-7b9d-459d-9006-65285f3efd23_narration_pass_2_69
        # the above was generated from egoclip.csv as follows:
        #     single_vid = df.iloc[j, 0] + '_' + df.iloc[j, 2] + '_' + str(df.iloc[j, 3])
        
        symlink_dir = "language_features/symlinks" # make this a self.symlink_dir on init function

        features_path = os.path.join(symlink_dir, video_filename)
        features = np.load(features_path, allow_pickle=True)
        features = torch.from_numpy(features).to(device=self.device) # note this disables gradients in some (maybe all) versions of pytorch

        # return before, after, cf1, cf2, cf3
        return features[0, 0, :], features[1, 0, :], features[2, 0, :], features[3, 0, :], features[4, 0, :], features[5, 0, :]

    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        before, after, cf1, cf2, cf3 = self._get_state_features(caption)
        final = self._get_video_frames(video_fp, video_sec, caption)

        # Scene-aware negative sampling
        if self.neg_param:
            # sample_neg = self.metadata[(self.metadata.video_uid==sample.video_uid)].sample(1).iloc[0] # variant of negative sample from same video
            sample_neg = self.metadata[self.metadata.segment_id==sample.segment_id].sample(1).iloc[0]
            video_fp_neg, video_sec_neg, bound_sec_neg = self._get_video_path(sample_neg)
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
            final_neg = self._get_video_frames(video_fp_neg, video_sec_neg, bound_sec_neg)

        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        # if self.neg_param:
        #     return {'video': final, 'text': caption,
        #             'video_neg': final_neg, 'text_neg': caption_neg,
        #             'meta': meta_arr,
        #             'noun_vec': noun_vec, 'verb_vec': verb_vec,
        #             'noun_vec_neg': noun_vec_neg, 'verb_vec_neg': verb_vec_neg}
        # else:
        #     return {'video': final, 'text': caption,
        #         'meta': meta_arr,
        #         'noun_vec': noun_vec, 'verb_vec': verb_vec}

        return {'video': final, 'text': caption,
                'meta': meta_arr,
                'before': before, 'after': after, 'cf1': cf1, 'cf2': cf2, 'cf3': cf3}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == 'train':
            return self._get_train_item(item)

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoClip_dataset",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="absolute/path/to/ego4d_chunked/",
        meta_dir="absolute/path/to/dataset/",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_egoclip',
        split='val',
        neg_param=60
    )
    dataset = EgoClip_EgoMCQ(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())

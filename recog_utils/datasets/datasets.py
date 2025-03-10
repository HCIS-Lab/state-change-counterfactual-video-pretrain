import os
import os.path
import numpy as np
import random
import torch
import json
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import re

# feature = "cf_18b_1e5_epoch5_correct"
# feature = "clip"
feature = "hiervl"
# feature = "milnce"

hidden_dim = 256

class Breakfast_feat(object):
    def __init__(self,
                 root='/nfs/wattrel/data/md0/datasets/action_seg_datasets/breakfast',
                 transform=None, mode='train',
                 num_frames=32, n_seg=64,
                 small_test=False,
                 frame_dir='/nfs/wattrel/data/md0/datasets/action_seg_datasets/breakfast/frames/',
                 class_dir="./data/id2acti.txt",
                 label_dir="./data/acti2id.json",
                 pretrain=True, n_split=5):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.n_seg = n_seg
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split
        
        print('#'*30)
        print("features: ", feature)
        print("split: ", n_split)
        print('#'*30)
        self.feat_dir = f"/nfs/wattrel/data/md0/datasets/action_seg_datasets/breakfast/{feature}_split{n_split}/combined_feat/"


        with open(self.label_dir, 'r') as f:
            self.cls2id = json.load(f)
            self.cls2id = {k: int(v) for k, v in self.cls2id.items()}
        self.classes = {}
        with open(self.class_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ', 1)
                self.classes[line[0]] = line[1]
                self.classes = {int(k): v for k, v in self.classes.items()}
        if self.mode == 'train':
            self.splt_dir = "./data/splits/train.split" + str(self.n_split) + ".bundle"
        else:
            self.splt_dir = "./data/splits/test.split" + str(self.n_split) + ".bundle"

        file_ptr = open(self.splt_dir, 'r')
        self.train_split = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        with open('./data/splits/breakfast_acti_vid2idx.json', 'r') as openf:
            self.act_splt = json.load(openf)

    def frame_sampler(self, vlen):
        if vlen > self.num_frames * self.n_seg:
            seq_idx = np.arange(0, vlen - self.num_frames, self.num_frames)
            seq_idx = np.append(seq_idx, vlen - self.num_frames)
            sorted_sample = [seq_idx[k] for k in sorted(random.sample(range(len(seq_idx)), self.n_seg))]
            result = [np.arange(ii, ii + self.num_frames) for ii in sorted_sample]
        else:
            result = [np.arange(i[0], i[0] + self.num_frames)
                      for i in np.array_split(range(vlen - self.num_frames), self.n_seg - 1)]
            result.append(np.arange(vlen - self.num_frames, vlen))

        return result
    
    @staticmethod
    def extract_frames(filename: str, num_frames=8):
        # Extract the number before ".npy" using regex
        match = re.search(r'_(\d+)\.npy$', filename)
        if not match:
            raise ValueError("Filename format is incorrect, expected a number before .npy")

        frame_idx = int(match.group(1))  # Convert to integer

        # Clean filename by removing the number before ".npy"
        cleaned_filename = re.sub(r'_\d+\.npy$', '.npy', filename)
        x = np.load(cleaned_filename)

        # Ensure we don't go out of bounds
        # T = x.shape[1]
        # start_idx = max(0, min(frame_idx, T - num_frames))  # Prevent overflow

        return x[:, frame_idx:frame_idx + num_frames].T
    
    @staticmethod
    def sample_evenly_spaced_frames(x, vid, window_size, num_samples=8):
        """
        Sample `num_samples` evenly spaced frames from a window of size `window_size`.

        Args:
            x: Tensor of shape [num_frames, feature_dim]
            vid: Starting frame index
            num_samples: Number of frames to sample (default: 8)
            window_size: Total range to sample frames from (default: 32)

        Returns:
            Tensor of shape [num_samples, feature_dim]
        """
        if feature == "clip":
            num_frames = x.shape[0]
        else:
            num_frames = x.shape[1]
        
        # Adjust window if it exceeds num_frames
        if vid + window_size > num_frames:
            window_size = num_frames - vid  # Reduce window size to fit in range
        
        # Generate `num_samples` evenly spaced indices within the valid window
        indices = torch.linspace(vid, vid + window_size - 1, steps=num_samples).long()
        # print(indices)
        
        if feature == "clip":
            return x[indices, :]
        else:
            return x[:, indices].T
    
    def __getitem__(self, index):

        videoname = self.train_split[index]
        videoname = videoname.split('.', 2)[0]
        vsplt = videoname.split('_')
        cls_id = self.cls2id[vsplt[3]]
        seq = np.zeros((self.n_seg, self.num_frames, hidden_dim))
        n = 0
        x = np.load(os.path.join(self.feat_dir, videoname + '.npy'))
        window_size = int(self.act_splt[videoname][1]) + int(self.act_splt[videoname][0])
        # print("x ", x.shape)
        for vid in self.act_splt[videoname]:
            # print("vid: ", vid)
            seq[n, :, :] = self.sample_evenly_spaced_frames(x=x, vid=int(vid), window_size=window_size)
             #self.extract_frames(os.path.join(self.feat_dir, videoname + '_' + vid + '.npy'), num_frames=self.num_frames)
            n += 1

        return seq, cls_id

    def __len__(self):
        # return 1
        return len(self.train_split)

class Breakfast_acti(data.Dataset):
    def __init__(self,
                 root='./data/breakfast',
                 transform=None, mode='val',
                 num_frames=32, ds=1, ol=0.5,
                 small_test=False,
                 frame_dir='/nfs/wattrel/data/md0/datasets/action_seg_datasets/breakfast/frames/',
                 label_dir='/nfs/wattrel/data/md0/datasets/activity_recognition/breakfast/action_ids/',
                 class_dir='/nfs/wattrel/data/md0/datasets/activity_recognition/breakfast/bf_mapping.json',
                 id2acti_dir="/nfs/wattrel/data/md0/datasets/activity_recognition/breakfast/id2acti.txt",
                 acti2id_dir="/nfs/wattrel/data/md0/datasets/activity_recognition/breakfast/acti2id.json",
                 pretrain=True, n_split=1):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split
        self.acti2id_dir = acti2id_dir
        self.id2acti_dir = id2acti_dir

        # if self.mode == 'train':
        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}

        with open(self.acti2id_dir, 'r') as f:
            self.cls2id = json.load(f)
            self.cls2id = {k: int(v) for k, v in self.cls2id.items()}
        self.acti_classes = {}
        with open(self.id2acti_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ', 1)
                self.acti_classes[line[0]] = line[1]
                self.acti_classes = {int(k): v for k, v in self.acti_classes.items()}
        # else:
        #     with open(self.ext_class_dir, 'r') as f:
        #         self.classes = json.load(f)
        #         self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split{self.n_split}_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}_all.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits', f'smalltest_split1_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        seq_idx = np.arange(self.num_frames) * self.ds + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0].split('_', 3)
        acti_name = vsplt[1]
        vd_name = vsplt[3]
        if acti_name == 'stereo':
            acti_name += '01'
            vd_name = vd_name[:-4]
        cls_id = self.cls2id[vd_name]
        vpath = os.path.join(self.frame_dir, vsplt[0], vsplt[1], vsplt[2] + '_' + vsplt[3])
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(
            os.path.join(self.label_dir, vsplt[0] + '_' + acti_name + '_' + vsplt[2] + '_' + vd_name + '.npy'))
        # diff = vlabel.size - vlen
        # if diff > 0:
        #     vlabel = vlabel[:-diff]
        # elif diff < 0:
        #     vlabel = np.pad(vlabel, (0, -diff), 'constant', constant_values=(0, vlabel[-1]))
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        vid = vlabel[frame_index]
        if self.pretrain:
            vid = torch.from_numpy(vid)
            vid = torch.unique_consecutive(vid)
            vid = vid.numpy()
            vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return seq, vid, cls_id

    def __len__(self):
        # return 1
        return len(self.train_split)
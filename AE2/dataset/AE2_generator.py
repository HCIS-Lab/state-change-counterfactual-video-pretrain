import os
import os.path
import numpy as np
import pickle

def _construct_video_path_by_mode(dir_name, mode):
    video_paths = []
    f_out = open(os.path.join(dir_name, mode + '.csv'), 'r')
    for line in f_out:
        line = line.strip()
        base_name = os.path.splitext(line)[0]  # Remove .mp4
        new_name = base_name + '_cf_epoch7.npy'
        video_paths.append(os.path.join(dir_name, new_name))
    return video_paths

def _construct_video_path(dir_name):
    video_paths = []
    for item in os.listdir(dir_name):
        if item.endswith('_cf_epoch7.npy'):
            video_paths.append(os.path.join(dir_name, item))
    assert len(video_paths) > 1
    print(f'{len(video_paths)} videos in {dir_name}')
    return video_paths

dataset = "pour_milk"
ego_only = False 

print()
print("dataset: ", dataset)
print("ego_only: ", ego_only)

if dataset in ["break_eggs", "tennis_forehand"]:
    for mode in ["train", "test", "val"]:
        print()
        print("mode: ", mode)

        ego_vid = _construct_video_path_by_mode(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/ego", mode)
        exo_vid = _construct_video_path_by_mode(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/exo", mode)

        if not ego_only:
            ego_vid += exo_vid

        with open(os.path.join(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}", 'label.pickle'), 'rb') as handle:
                    dircy = pickle.load(handle)
        feats = []
        labels = []
        for i in ego_vid:
            x = np.load(i)
            video_name = i.replace('_cf_epoch7.npy', '').split('/')[-1]
            label = dircy[video_name]
            # print(torch.from_numpy(x).shape)
            feats.append(x)
            labels.append(label)


        feats = np.concatenate(feats, axis=-1)
        labels = np.concatenate(labels)
        feats0 = feats.T.copy()
        print("feats: ", feats.shape)
        print("feats0: ", feats0.shape)
        print("labels: ", labels.shape)
        print('-'*20)
else:
    for mode in ["train", "test", "val"]:
        print()
        print("mode: ", mode)

        ego_vid = _construct_video_path(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/{mode}/ego")
        exo_vid = _construct_video_path(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/{mode}/exo")

        if not ego_only:
            ego_vid += exo_vid

        with open(os.path.join(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}", 'label.pickle'), 'rb') as handle:
                    dircy = pickle.load(handle)
        feats = []
        labels = []
        for i in ego_vid:
            x = np.load(i)
            video_name = i.replace('_cf_epoch7.npy', '').split('/')[-1]
            label = dircy[video_name]
            # print(torch.from_numpy(x).shape)
            feats.append(x)
            labels.append(label)


        feats = np.concatenate(feats, axis=-1)
        labels = np.concatenate(labels)
        feats0 = feats.T.copy()
        print("feats: ", feats.shape)
        print("feats0: ", feats0.shape)
        print("labels: ", labels.shape)
        print('-'*20)


if not ego_only:
    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds.npy", feats0)
    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label.npy", labels)
else:
    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_egoOnly.npy", feats0)
    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_egoOnly.npy", labels)

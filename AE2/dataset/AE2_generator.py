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
        new_name = base_name + '_milnce.npy'
        video_paths.append(os.path.join(dir_name, new_name))
    return video_paths

def _construct_video_path(dir_name):
    video_paths = []
    for item in os.listdir(dir_name):
        if item.endswith('_milnce.npy'):
            video_paths.append(os.path.join(dir_name, item))
    assert len(video_paths) > 1
    print(f'{len(video_paths)} videos in {dir_name}')
    return video_paths

# dataset = "pour_milk"
# ego_only = False 
clipp = False
for dataset in ["break_eggs", "pour_milk", "pour_liquid", "tennis_forehand"]:
    print()
    print('#'*20)
    print("dataset: ", dataset)
    for ego_only in [True, False]:
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
                    if clipp:
                        x = x.T
                    video_name = i.replace('_milnce.npy', '').split('/')[-1]
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
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_milnce.npy", feats0)
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_milnce.npy", labels)
                else:
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_egoOnly_milnce.npy", feats0)
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_egoOnly_milnce.npy", labels)

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
                    if clipp:
                        x = x.T
                    video_name = i.replace('_milnce.npy', '').split('/')[-1]
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
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_milnce.npy", feats0)
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_milnce.npy", labels)
                else:
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_egoOnly_milnce.npy", feats0)
                    np.save(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_egoOnly_milnce.npy", labels)

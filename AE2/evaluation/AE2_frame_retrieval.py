import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def _construct_video_path_by_mode(dir_name, mode):
    video_paths = []
    f_out = open(os.path.join(dir_name, mode + '.csv'), 'r')
    for line in f_out:
        line = line.strip()
        base_name = os.path.splitext(line)[0]  # Remove .mp4
        new_name = base_name + '_clip.npy'
        video_paths.append(os.path.join(dir_name, new_name))
    return video_paths

def _construct_video_path(dir_name):
    video_paths = []
    for item in os.listdir(dir_name):
        if item.endswith('_clip.npy'):
            video_paths.append(os.path.join(dir_name, item))
    assert len(video_paths) > 1
    print(f'{len(video_paths)} videos in {dir_name}')
    return video_paths

def retrieval_ap_at_k(video_len_list, video_paths, embeddings, labels, k_list, cross_view=False):
    dim = 2 if cross_view else 1
    ap = np.zeros((len(k_list), dim))
    num_queries = np.zeros((len(k_list), dim))

    nbrs = NearestNeighbors(n_neighbors=embeddings.shape[0], algorithm='auto').fit(embeddings)
    # nbrs = NearestNeighbors(n_neighbors=k_list[-1] * 50, algorithm='auto').fit(embeddings)

    frameid2videoid = {}
    cur_idx = 0
    for i, video_file in enumerate(video_paths):
        video_len = video_len_list[i]
        is_ego = True if 'ego' in video_file else False
        for frameid in range(cur_idx, cur_idx + video_len):
            frameid2videoid[frameid] = [i, is_ego, frameid - cur_idx]
        cur_idx = cur_idx + video_len

    for i in tqdm(range(embeddings.shape[0])):
        # Get the query embedding and label
        query_embedding = embeddings[i]
        query_label = labels[i]

        # Find the K+1 nearest neighbors (the first neighbor is the query it
        distances, indices = nbrs.kneighbors([query_embedding])
        indices = indices.flatten()

        if cross_view:
            indices = [j for j in indices if
                       frameid2videoid[j][0] != frameid2videoid[i][0]
                       and frameid2videoid[j][1] != frameid2videoid[i][1]]
        else:
            indices = [j for j in indices if frameid2videoid[j][0] != frameid2videoid[i][0]]

        for k_idx, k in enumerate(k_list):
            indices_s = indices[:k].copy()
            assert len(indices_s) == k

            # Count the number of relevant neighbors (with the same label as the query)
            num_relevant = np.sum(labels[indices_s] == query_label)

            # Calculate precision at each rank up to K
            precisions = np.zeros(k)
            for j in range(k):
                precisions[j] = np.sum(labels[indices_s[:j + 1]] == query_label) / (j + 1)

            # Calculate average precision for this query
            if cross_view:
                ego_idx = int(frameid2videoid[i][1])
            else:
                ego_idx = 0
            if num_relevant > 0:
                ap[k_idx][ego_idx] += np.sum(precisions * (labels[indices_s] == query_label)) / num_relevant
            else:
                ap[k_idx][ego_idx] += 0.0
            num_queries[k_idx][ego_idx] += 1

    if cross_view:
        ego2exo = (ap / num_queries)[:, 1]
        exo2ego = (ap / num_queries)[:, 0]
        return ego2exo.squeeze(), exo2ego.squeeze()

    else:
        return (ap / num_queries).squeeze()


def frame_retrieval(dataset, video_len_list, video_paths):
    train_embs = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/train_embeds_clip.npy")
    train_labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/train_label_clip.npy")
    
    val_embs = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/val_embeds_clip.npy")
    val_labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/val_label_clip.npy")

    test_embs = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/test_embeds_clip.npy")
    test_labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/test_label_clip.npy")

    te = np.concatenate([train_embs, test_embs, val_embs], axis=0)
    tl = np.concatenate([train_labels, test_labels, val_labels], axis=0)


    regular = retrieval_ap_at_k(video_len_list, video_paths, te, tl, [10], cross_view=False)
    # ego2exo, exo2ego = retrieval_ap_at_k(video_len_list, video_paths, val_embs, val_labels, [10], cross_view=True)
    return regular#, ego2exo, exo2ego

def main():
    md ='clip'
    print(md)

    # mode = 'test'
    print("mode: ", 'train + test + val')
    sets = ["break_eggs", "pour_milk", "pour_liquid", "tennis_forehand"]
    avgs = {"ego_and_exo": [], "ego2exo": [], "exo2ego":[], "ego_only": []}
    
    for dataset in sets:
        print()
        print("-"*20)
        print("dataset: ", dataset)

        video_len_list = []
        print()
        ego_vid = []
        # exo_vid = []
        for mode in ['train', 'test', 'val']:
            if dataset in ["break_eggs", "tennis_forehand"]:
                ego_vid += _construct_video_path_by_mode(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/ego", mode)
                ego_vid += _construct_video_path_by_mode(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/exo", mode)
            else:
                ego_vid += _construct_video_path(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/{mode}/ego")
                ego_vid += _construct_video_path(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/{dataset}/{mode}/exo")
            
            # ego_vid += exo_vid

        print("len before: ", len(ego_vid))
        ego_vid = set(ego_vid)
        print("len after: ", len(ego_vid))

        for video in ego_vid:
            video_frames_count = int(np.load(video).shape[-1])
            # video_frames_count = int(np.load(video).shape[0])
            video_len_list.append(video_frames_count)
        
        reg_map10 = frame_retrieval(dataset, video_len_list, ego_vid)

        avgs["ego_and_exo"].append(reg_map10)
        # avgs["ego2exo"].append(ego2exo_F1)
        # avgs["exo2ego"].append(exo2ego_F1)
        # avgs["ego_only"].append(ego_onlyF1)

        print(f'Test MAP@10: regular_all={reg_map10:.4f}')# | ego2exo={ego2exo_F1:.4f} | exo2ego={exo2ego_F1:.4f} | ego_only={ego_onlyF1:.4f}')
    
    print()
    for k,v in avgs.items():
        print(k, sum(v) / 4)

if __name__ == "__main__":
    main()


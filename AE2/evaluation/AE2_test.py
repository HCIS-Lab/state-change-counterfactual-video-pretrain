import random
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
np.random.seed(0)
random.seed(0)

def fit_svm_model(train_embs, train_labels, test_embs, test_labels, cal_f1_score=False):
    """Fit a SVM classifier."""
    svm_model = SVC(decision_function_shape='ovo', verbose=False)
    svm_model.fit(train_embs, train_labels)

    test_preds = svm_model.predict(test_embs)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    return test_f1

def classification(dataset):
    # ego and exo
    train_embs, train_labels = load_data(dataset=dataset, mode='train', ego_only=False, exo_only=False)
    val_embs, val_labels = load_data(dataset=dataset, mode='val', ego_only=False, exo_only=False)
    test_embs, test_labels = load_data(dataset=dataset, mode='test', ego_only=False, exo_only=False)    
    te = np.concatenate([test_embs, val_embs], axis=0)
    tl = np.concatenate([test_labels, val_labels], axis=0)
    # te=test_embs
    # tl=test_labels

    regular_f1 = fit_svm_model(train_embs, train_labels, te, tl, cal_f1_score=True)
    del train_embs, train_labels, val_embs, val_labels, test_embs, test_labels, te, tl

    # ego2exo
    train_embs, train_labels = load_data(dataset=dataset, mode='train', ego_only=True, exo_only=False)
    val_embs, val_labels = load_data(dataset=dataset, mode='val', ego_only=False, exo_only=True)
    test_embs, test_labels = load_data(dataset=dataset, mode='test', ego_only=False, exo_only=True)
    te = np.concatenate([test_embs, val_embs], axis=0)
    tl = np.concatenate([test_labels, val_labels], axis=0)
    # te=test_embs
    # tl=test_labels

    ego2exo_F1 = fit_svm_model(train_embs, train_labels, te, tl, cal_f1_score=True)
    del train_embs, train_labels, val_embs, val_labels, test_embs, test_labels, te, tl

    # exo2ego
    train_embs, train_labels = load_data(dataset=dataset, mode='train', ego_only=False, exo_only=True)
    val_embs, val_labels = load_data(dataset=dataset, mode='val', ego_only=True, exo_only=False)
    test_embs, test_labels = load_data(dataset=dataset, mode='test', ego_only=True, exo_only=False)
    te = np.concatenate([test_embs, val_embs], axis=0)
    tl = np.concatenate([test_labels, val_labels], axis=0)
    # te=test_embs
    # tl=test_labels

    exo2ego_F1 = fit_svm_model(train_embs, train_labels, te, tl, cal_f1_score=True)
    del train_embs, train_labels, val_embs, val_labels, test_embs, test_labels, te, tl

    # ego only
    train_embs, train_labels = load_data(dataset=dataset, mode='train', ego_only=True, exo_only=False)
    val_embs, val_labels = load_data(dataset=dataset, mode='val', ego_only=True, exo_only=False)
    test_embs, test_labels = load_data(dataset=dataset, mode='test', ego_only=True, exo_only=False)
    te = np.concatenate([test_embs, val_embs], axis=0)
    tl = np.concatenate([test_labels, val_labels], axis=0)
    # te=test_embs
    # tl=test_labels

    ego_onlyF1 = fit_svm_model(train_embs, train_labels, te, tl, cal_f1_score=True)
    del train_embs, train_labels, val_embs, val_labels, test_embs, test_labels, te, tl

    return regular_f1, ego2exo_F1, exo2ego_F1, ego_onlyF1

def load_data(dataset, mode, ego_only, exo_only):
    if not ego_only and not exo_only: # if not ego_only and not ego_only:
        embeds = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_ablation1.npy")
        labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_ablation1.npy")
    elif not ego_only and exo_only:
        embeds = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_exoOnly_ablation1.npy")
        labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_exoOnly_ablation1.npy")
    elif ego_only and not exo_only:
        embeds = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_embeds_egoOnly_ablation1.npy")
        labels = np.load(f"/nfs/wattrel/data/md0/datasets/AE2/AE2_data/embeddings/{dataset}/{mode}_label_egoOnly_ablation1.npy")
    else:
        raise NotImplementedError("This mode is not available yet.")
    
    return embeds, labels

def main():
    print("ablation1")

    sets = ["break_eggs", "pour_milk", "pour_liquid", "tennis_forehand"]
    avgs = {"ego_and_exo": [], "ego2exo": [], "exo2ego":[], "ego_only": []}
    for dataset in sets:
        print()
        print("-"*20)
        print("dataset: ", dataset)
        regular_f1, ego2exo_F1, exo2ego_F1, ego_onlyF1 = classification(dataset=dataset)
        avgs["ego_and_exo"].append(regular_f1)
        avgs["ego2exo"].append(ego2exo_F1)
        avgs["exo2ego"].append(exo2ego_F1)
        avgs["ego_only"].append(ego_onlyF1)

        print(f'Test F1: ego_and_exo={regular_f1:.4f} | ego2exo={ego2exo_F1:.4f} | exo2ego={exo2ego_F1:.4f} | ego_only={ego_onlyF1:.4f}')
    
    print()
    for k,v in avgs.items():
        print(k, sum(v) / 4)
if __name__ == "__main__":
    main()
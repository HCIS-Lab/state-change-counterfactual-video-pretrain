ckpt_dir=/nfs/wattrel/data/md0/datasets/AE2/AE2_data

# extract embeddings then evaluate
# break eggs
python evaluation/evaluate_features.py --dataset break_eggs --task align \
    --extract_embedding \
    --hidden_dim 768 --n_layers 1 \
    --eval_task 1 \
    --ckpt $ckpt_dir

# pour milk, missing one det_bounding_box.pickle file, to be updated later
python evaluation/evaluate_features.py --dataset pour_milk --task align \
   --extract_embedding \
   --hidden_dim 768 --n_layers 1 \
   --eval_task 1 \
   --ckpt $ckpt_dir

# pour liquid
python evaluation/evaluate_features.py --dataset pour_liquid --task align \
    --extract_embedding \
    --use_bbox_pe --weigh_token_by_bbox \
    --hidden_dim 768 --n_layers 3 \
    --eval_task 1 \
    --ckpt $ckpt_dir

# tennis forehand
python evaluation/evaluate_features.py --dataset tennis_forehand --task align \
    --use_bbox_pe --weigh_token_by_bbox --use_mask --one_object_bbox \
    --hidden_dim 768 --n_layers 1 \
    --eval_task 1 \
    --ckpt $ckpt_dir


# python evaluation/evaluate_features.py --dataset pour_milk --task align --extract_embedding --hidden_dim 500 --n_layers 1 --eval_task 1234 --ckpt /nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0226_23_46_03/ckpt_18b_1e5_epoch7_correct.pth
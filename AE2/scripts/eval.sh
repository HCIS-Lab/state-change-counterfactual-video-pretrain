ckpt_dir=/nfs/wattrel/data/md0/datasets/AE2/AE2_data

# use extracted embeddings (saved in ckpt_dir/$dataset_eval)
# for dataset in break_eggs; do
#     python evaluation/evaluate_features.py --dataset $dataset \
#         --eval_task 1 \
#         --ckpt $ckpt_dir/$dataset

python evaluation/evaluate_features.py --dataset break_eggs --task align \
    --extract_embedding \
    --hidden_dim 256 --n_layers 1 \
    --eval_task 1234 \
    --ckpt $ckpt_dir
done
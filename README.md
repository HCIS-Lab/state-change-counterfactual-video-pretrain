# What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning

Official code of [What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning]([https://openaccess.thecvf.com/content/CVPR2023/html/Ashutosh_HierVL_Learning_Hierarchical_Video-Language_Embeddings_CVPR_2023_paper.html](https://arxiv.org/abs/2503.21055)), ICCV 2025.

## Installation

To create a conda environment with the required dependencies, run the following command:

```bash
conda env create -f environment.yml
source activate cf
```

## Dataset Preparation

### Pre-training

Ego4D/EgoClip

Please refer to [EgoVLP](https://github.com/showlab/EgoVLP) codebase for data preparation. We use the downsampled and chunked video outputs as the input to our method (output from `utils/video_chunk.py`). For summary sentences, we provide the processed summary and narration hierarchy [here](https://dl.fbaipublicfiles.com/hiervl/summary_clips_hierarchy_full.json). The used `egosummary_full.csv` is available [here](https://dl.fbaipublicfiles.com/hiervl/egosummary_full.csv).

### Downstream tasks

GTEA: Please follow [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt) to download the raw video and then extract frames from videos.

EgoPRE: Link.

EpicKitchen & Charades-Ego: Please refer to [EgoVLP](https://github.com/showlab/EgoVLP) codebase for data preparation.

AE2:

## Generate State Changes and Their Counterfactuals with Llama

Please refer to [Llama 3](https://github.com/meta-llama/llama3) for model weights and instructions. We use the scripts to generate state change and counterfactual descriptions for the entire Ego4D dataset:

```
# clip-level state changes and their counterfactuals
cd llama_script
python clip_level_sc_cf.py

# video-level counterfactuals
cd llama_script
python video_level_cf.py
```


## Generate Text Features with FLAVA

## Pretraining

We use two nodes for distributed training. Each node has 4 32GB GPUs. The pretraining can be run as

```bash
python -m torch.distributed.launch  --nnodes=$HOST_NUM  --node_rank=$INDEX  --master_addr $CHIEF_IP  --nproc_per_node $HOST_GPU_NUM  --master_port 8081  run/train_egoaggregate.py --config configs/pt/egoaggregation.json
```

We experiment mainly on SLURM and the instructions to run this code on SLURM is given next.

## Running on SLURM cluster

To run the pretraining on a distributed SLURM system, copy the content of `slurm_scripts` to this directly level and run

```
bash mover_trainer.sh job_name
```

The parameters of the SLURM job can be changed in the trainer.sh script. We use 2 nodes, each with 4 32 GB GPUs. The submit schedule first copies the required scripts to a different folder and then runs it from there. This copying ensures the code can be safely edited while a job is in the SLURM queue.

## Pretraining Checkpoint

The pretraining checkpoint is available [here](https://drive.google.com/drive/folders/1fNGuHmyzqygvgbvvE07GylB90kt-NlTi).


## Downstream Task Training

### Temporal Action Segmentation

### Error Detection

### AE2 Action Phase Recognition

## Downstream Task Testing

### Temporal Action Segmentation

### Error Detection

### EpicKitchen-100 Zero-Shot Multi-Instance Retrieval

```python
python downstream_script/test_epic.py
```

### Charades-Ego Zero-Shot Action Classification

### AE2 Zero-Shot Action Phase Frame Retrieval


To test the performance, run

```python
python run/test_charades.py
```


## Citation

If you use the code or the method, please cite the following paper:

```bibtek
@InProceedings{counterfactual_ICCV_2025,
    author    = {Kung, Chi-Hsi and Ramirez, Frangil and Ha, Juhyung and Chen, Yi-Ting and Crandall, David and Tsai, Yi-Hsuan},
    title     = {What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```

## Acknowledgement

The pretraining and Chrades-Ego, EPIC-KITCHENS test codebase is based on [EgoVLP](https://github.com/showlab/EgoVLP) and [HierVL](https://github.com/facebookresearch/HierVL). 

The feature extraction code is adapted from [Bridge-Prompt](https://github.com/ttlmh/Bridge-Prompt).

The temporal action segmentation code is adapted from [ASFormer](https://github.com/ChinaYi/ASFormer).

The action phase recognition and frame retrieval code is adapted from [AE2]()

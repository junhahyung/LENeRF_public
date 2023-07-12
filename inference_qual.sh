#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-0}
N_PROC=$((${#CUDA_VISIBLE_DEVICES}/2+1))
python -m torch.distributed.launch --nproc_per_node=$N_PROC --master_port=$RANDOM infer_qual.py  --noise_mode const --N_images 5  

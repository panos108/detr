#!/bin/bash

#$ -N train_detr_gpu
#$ -cwd
#$ -q gpu-a100rnd
#$ -l gpus=1
#$ -l gpumemory=20G

SCRIPT_PATH=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/train.py 
TRAIN_DATA=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/train/ 
TRAIN_ANN=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/train/_annotations.coco.json 
VAL_DATA=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/valid/ 
VAL_ANN=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/valid/_annotations.coco.json 

/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/env/bin/python $SCRIPT_PATH $TRAIN_DATA $TRAIN_ANN $VAL_DATA $VAL_ANN --num_devices 1 --batch_size 2 --max_epochs 400 --accelerator gpu --gradient_clip_val 0.1 --early_stop_patience 80
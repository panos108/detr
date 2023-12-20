#!/bin/bash

#$ -N train_detr
#$ -cwd
#$ -pe threaded 8
#$ -l h_vmem=64G

SCRIPT_PATH=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/train.py 
TRAIN_DATA=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/train/ 
TRAIN_ANN=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/train/_annotations.coco.json 
VAL_DATA=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/valid/ 
VAL_ANN=/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/DETR/AquariumDetection/valid/_annotations.coco.json 

/illumina/scratch/Titan_Isilon_UK/Users/tnuntasukk/env/bin/python $SCRIPT_PATH $TRAIN_DATA $TRAIN_ANN $VAL_DATA $VAL_ANN --num_devices 8 --batch_size 4 --max_epochs 400 --lr 0.001 --lr_backbone 0.0001
import os
import src
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

parser = argparse.ArgumentParser()

parser.add_argument("train_dir_path", type=str)
parser.add_argument("train_ann_path", type=str)
parser.add_argument("val_dir_path", type=str)
parser.add_argument("val_ann_path", type=str)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_backbone", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--min_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--early_stop_patience", type=int, default=50)
parser.add_argument("--accelerator", type=str, default="cpu")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--gradient_clip_val", type=str, default="None")

args = parser.parse_args()

# Setting all the random seeds to the same value.
# Each rank will get its own set of initial weights, if they don't match up, the gradients will not match
# leading to training that may not converge.
pl.seed_everything(1)

train_dataset = src.CocoDetection(img_folder=args.train_dir_path,
                                  ann_file=args.train_ann_path)

val_dataset = src.CocoDetection(img_folder=args.val_dir_path,
                                ann_file=args.val_ann_path)

train_cat = train_dataset.coco.cats
val_cat = val_dataset.coco.cats
all_cat = train_cat | val_cat
n_labels = len(all_cat)

gradient_clip_val = float(args.gradient_clip_val) if (args.gradient_clip_val != "None") else None

accelerator = args.accelerator if args.accelerator in ["cpu", "gpu"] else "auto"
num_devices = args.num_devices

# Set number of workers to number of cpus
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=train_dataset.collate_fn, num_workers=num_devices)

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=val_dataset.collate_fn, num_workers=num_devices)

model = src.Detr(num_labels=n_labels,
                 lr=args.lr,
                 lr_backbone=args.lr_backbone,
                 weight_decay=args.weight_decay)

wandb_logger = WandbLogger(project="DetrAquarium", log_model="all", entity="sei-uk")
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=0, patience=args.early_stop_patience)
model_callback = ModelCheckpoint(monitor="val_loss", mode="min")

trainer = pl.Trainer(log_every_n_steps=1,
                     devices=num_devices,
                     logger=wandb_logger,
                     accelerator=accelerator,
                     min_epochs=args.min_epochs,
                     max_epochs=args.max_epochs,
                     gradient_clip_val=gradient_clip_val,
                     callbacks=[early_stop_callback, model_callback])

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

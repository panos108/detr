import os
import wandb
import torch
import torchvision
import pytorch_lightning as pl
from PIL import Image, ImageDraw
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def __getitem__(self, idx):
        # Get PIL image and target dict in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        # pixel_values = img

        # Preprocess image and target (Format target dict for DETR, Resize & Normalize image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")

        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

    def collate_fn(self, batch):
        labels = [item[1] for item in batch]
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")

        batch = {"pixel_values": encoding["pixel_values"],
                 "pixel_mask": encoding["pixel_mask"],
                 "labels": labels
                 }

        return batch


class Detr(pl.LightningModule):

    def __init__(self, num_labels, lr, lr_backbone, weight_decay):

        super().__init__()
        self.save_hyperparameters()

        # replace COCO classification head with custom head (specify num_labels)
        # specify "no_timm" to not rely on the timm library for the backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=num_labels,
                                                            ignore_mismatched_sizes=True)

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        # labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        self.log("train_loss", loss, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        self.log("val_loss", loss, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item(), sync_dist=True)

        return loss

    def configure_optimizers(self):
        param_dicts = \
            [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                 "lr": self.lr_backbone}
            ]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def on_validation_epoch_end(self):
        val_dataset = self.trainer.val_dataloaders.dataset

        cats = val_dataset.coco.cats
        id2label = {k: v['name'] for k, v in cats.items()}
        table = wandb.Table(columns=["Image ID", "Prediction"])

        for i in range(len(val_dataset)):
            pixel_values, target = val_dataset[i]
            pixel_values = pixel_values.unsqueeze(0)

            # outputs: dict_keys(['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'])
            outputs = self.model(pixel_values=pixel_values, pixel_mask=None)

            # Retrieve image & annotations
            image_id = target['image_id'].item()
            image = val_dataset.coco.loadImgs(image_id)[0]
            annotations = val_dataset.coco.imgToAnns[image_id]
            image = Image.open(os.path.join('AquariumDetection/valid', image['file_name']))
            width, height = image.size

            # postprocess model outputs (return list of dicts (scores, labels, boxes) for images in the batch)
            # box format: (xmin, ymin, xmax, ymax)
            processed_outputs = self.processor.post_process_object_detection(outputs,
                                                                             target_sizes=[(height, width)],
                                                                             threshold=0.5)
            results = processed_outputs[0]
            scores = results["scores"].tolist()
            labels = results["labels"].tolist()
            boxes = results["boxes"].tolist()

            pred_boxes = []
            truth_boxes = []

            # prediction boxes
            for score, label, box_pos in zip(scores, labels, boxes):
                xmin, ymin, xmax, ymax = box_pos

                box = {"position": {"minX": xmin, "maxX": xmax, "minY": ymin, "maxY": ymax},
                       "class_id": label,
                       "scores": {"Logit": score},
                       "domain": "pixel",
                       "box_caption": f"{id2label[label]}: {score: .2f}"}

                pred_boxes.append(box)

            # ground_truth boxes
            for annotation in annotations:
                xmin, ymin, w, h = tuple(annotation['bbox'])
                class_id = annotation['category_id']

                box = {"position": {"minX": xmin, "maxX": xmin+w, "minY": ymin, "maxY": ymin+h},
                       "class_id": class_id,
                       "domain": "pixel",
                       "box_caption": id2label[class_id]}

                truth_boxes.append(box)

            box_img = wandb.Image(image,
                                  boxes={"predictions": {"box_data": pred_boxes,
                                                         "class_labels": id2label},

                                         "ground_truth": {"box_data": truth_boxes,
                                                          "class_labels": id2label}})
            table.add_data(image_id, box_img)

        self.logger.experiment.log({"val_prediction": table}, )



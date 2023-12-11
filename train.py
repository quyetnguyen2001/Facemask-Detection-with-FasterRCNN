from dataset import *
import os
import argparse
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import shutil


def get_args():
    parser = argparse.ArgumentParser(description="train object detector")
    parser.add_argument("--batch-size", "-b", type=int, default=3)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--image-size", "-i", type=int, default=416)

    args = parser.parse_args()
    return args


args = get_args()


def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.Resize(args.image_size, args.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), 
    ToTensorV2()],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

train_dataset = Face_mask_dataset(transform=transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=4, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4*4, bias=True)
model.to(device)


if os.path.isdir("tensorboard"):
    shutil.rmtree("tensorboard")
os.mkdir("tensorboard")
writer = SummaryWriter("tensorboard")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_loss = 100

for epoch in range(0,args.epochs):
    model.train()
    train_loss = []
    progressbar = tqdm(train_dataloader)
    for i, (images,targets) in enumerate(progressbar):
        images = [image.to(device) for image in images]
        targets = [{'boxes':  target['boxes'].to(device), 'labels': target['labels'].to(device)} for target in targets]
        output = model(images,targets)
        loss = sum([l for l in output.values()])
        optimizer.zero_grad()
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        progressbar.set_description("Epoch {}. Iteration {}/{}. Loss {}".format(epoch+1, i+1, len(train_dataloader), np.mean(train_loss)))
        writer.add_scalar("Train/Loss", np.mean(train_loss), epoch)
    checkpoint = {
        "epoch": epoch+1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join("model", "faster_last.pt"))
    if np.mean(train_loss) < best_loss:
        best_loss = np.mean(train_loss)
        torch.save(checkpoint, os.path.join("model", "faster_best.pt"))
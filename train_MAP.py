from dataset import *
import argparse
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import albumentations as A
from pprint import pprint
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torchvision
from tqdm.autonotebook import tqdm
from torchmetrics.detection import MeanAveragePrecision

def get_args():
    parser = argparse.ArgumentParser(description="Map of model")
    parser.add_argument("--model-path", "-m", type=str,
                        default="model/faster_best.pt")
    parser.add_argument("--size", "-s", type=int, default=416)
    parser.add_argument("--image-size", "-is", type=int, default=416)
    parser.add_argument("--batch-size", "-b", type=int, default=3)
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
    A.Resize(args.image_size, args.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), 
    ToTensorV2()],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

test_dataset = Face_mask_dataset(transform=transform)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=4, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4*4, bias=True)
model_path = torch.load(args.model_path)
model.load_state_dict(model_path["model"])
model.to(device)
model.eval()
metric = MeanAveragePrecision(class_metrics=True)
val_progress_bar = tqdm(test_dataloader, colour="yellow")
for i, (images, targets) in enumerate(val_progress_bar):
    images = list(image.to(device) for image in images)
    targets_list = []
    for t in targets:
        target = {}
        target["boxes"] = t["boxes"].to(device)
        target["labels"] = t["labels"].to(device)
        targets_list.append(target)
    with torch.no_grad():
        outputs = model(images)
    output = outputs[0]
    target_1 = targets_list[0]
    preds = [{'boxes':output["boxes"], "scores": output["scores"],'labels':output["labels"]}]
    gts = [{'boxes':target_1["boxes"] , 'labels':target_1["labels"]}]
    metric.update(preds, gts)
map_dict = metric.compute()
pprint(map_dict)

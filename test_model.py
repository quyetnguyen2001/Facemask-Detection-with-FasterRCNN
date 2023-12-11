import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument("--model-path", "-m", type=str,
                        default="model/faster_best.pt")
    parser.add_argument("--image", "-i", type=str, default=None)
    parser.add_argument("--threshold", "-t", type=float, default=0.75)
    args = parser.parse_args()
    return args


args = get_args()
if torch.cuda.is_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classes = ["background", "mask", "mask_weared_incorrect", "no_mask"]
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor.cls_score = nn.Linear(
    in_features=1024, out_features=4, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(
    in_features=1024, out_features=4*4, bias=True)
model.to(device)

model_path = torch.load(args.model_path)
model.load_state_dict(model_path["model"])

model.eval()

image = Image.open(args.image).convert("RGB")
width, height = image.size
transform_image = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]
)
test_image = transform_image(image)
test_image = test_image.unsqueeze(dim=0).to(device)


with torch.no_grad():
    outputs = model(test_image)

output = outputs[0]

boxes = output['boxes']

labels = output['labels']
scores = output['scores']
final_boxes = []
final_labels = []
final_scores = []
for b, l, s in zip(boxes, labels, scores):
    if s > args.threshold:
        final_scores.append(s)
        final_boxes.append(b)
        final_labels.append(l)
image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for b, l in zip(final_boxes, final_labels):
    xmin, ymin, xmax, ymax = b
    xmin = int(xmin / args.image_size * width)
    ymin = int(ymin / args.image_size * height)
    xmax = int(xmax / args.image_size * width)
    ymax = int(ymax / args.image_size * height)

    if l == 1:
        image = cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(
            classes[l], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        image = cv2.rectangle(image, (xmin, ymin - 20),
                              (xmin + w, ymin), (0, 255, 0), -1)
        image = cv2.putText(image, classes[l], (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        image = cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (255, 0, 0), 2)
        (w, h), _ = cv2.getTextSize(
            classes[l], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        image = cv2.rectangle(image, (xmin, ymin - 20),
                              (xmin + w, ymin), (255, 0, 0), -1)
        image = cv2.putText(image, classes[l], (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)


cv2.imshow('output', image)
cv2.waitKey(0)

# [PYTORCH] Faster R-CNN
## Introduction
I practicing pytorch with pretrained Faster-RCNN model. I trained the model with dataset has been label on the [kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

<p align="center">
  <img width="650" height="450" src=output.jpg>
</p>
<p align="center">
  <img width="650" height="450" src=output1.jpg>
</p>


<p align="center">
Examples of my model's output.
</p>

## Requirements: 
- **python 3.10**
- **torch 2.0**
- **opencv (cv2)**
- **numpy**
- **matplotlib**
- **argparse**

## Mean Average Precision 
<p align="center">
  <img width="650" height="250" src=train_MAP.png>
</p>
<p align="center">
Mean Average Precision in 10 epochs.
</p>

## How to run my code:
```c
python test_model.py --image  image.jpg     # image
```

import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
import cv2


class Face_mask_dataset(Dataset):
    def __init__(self, root="dataset", transform=None):
        self.root = root
        self.transform = transform
        self.classes = ["background", "with_mask",
                        "mask_weared_incorrect", "without_mask"]

    def __len__(self):
        image_path = os.listdir(os.path.join(self.root, "images"))
        return len(image_path)

    def __getitem__(self, index):
        xml_path = os.path.join(self.root, "annotations")
        xml_path_list = os.listdir(xml_path)
        xml_file = os.path.join(xml_path, xml_path_list[index])

        anot = ET.parse(xml_file)
        name = anot.find("filename").text
        image_path = os.path.join(self.root, "images", name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []
        labels = []
        for obj in anot.findall("object"):
            xmin, ymin, xmax, ymax = [int(obj.find("bndbox").find(
                tag).text)-1 for tag in ["xmin", "ymin", "xmax", 'ymax']]
            bboxes.append([xmin, ymin, xmax, ymax])
            label = self.classes.index(obj.find("name").text.lower().strip())
            labels.append(label)

        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        targets = {"boxes": bboxes, "labels": labels}
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels = labels)
            image = transformed['image']
            targets['boxes'] = torch.tensor(transformed['bboxes'])

        return image,targets

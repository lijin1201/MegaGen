import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, df, img_size=(256, 256), augment=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

        if augment:
            self.transform = A.Compose([
                A.Rotate(limit=0.2 * 180),  # degrees
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0),
                A.HorizontalFlip(p=0.5),
                A.Affine(shear={"x": 3, "y": 3}),
                A.Resize(*img_size),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*img_size),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['images_paths'])  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row['masks_paths'], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)  # binarize

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)  # [1, H, W]

        return image, mask

def create_dataloader(df, img_size=(256, 256), batch_size=40, shuffle=True, augment=True, num_workers=4):
    dataset = SegmentationDataset(df, img_size=img_size, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def show_images(images, masks):
    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        img_path = images[i]
        mask_path = masks[i]
        # read image and convert it to RGB scale
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read mask
        mask = cv2.imread(mask_path)
        # sho image and mask
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)

        plt.axis('off')

    plt.tight_layout()
    plt.show()


def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = y_true.view(-1)
    y_pred_flatten = y_pred.view(-1)

    intersection = (y_true_flatten * y_pred_flatten).sum()
    union = y_true_flatten.sum() + y_pred_flatten.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

def iou_coef(y_true, y_pred, smooth=100):
    intersection = (y_true * y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
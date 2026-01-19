from cv2 import transform
import torch
import torchvision.transforms.functional as tf

import numpy as np

class SimSiamDataset(torch.utils.data.Dataset) :
    def __init__(self, base_dataset, transform = None) :
        self.transform = transform
        self.base_dataset = base_dataset
        self.len = len(base_dataset)

    def __len__(self) :
        return self.len

    def __getitem__(self, idx) :
        x, _ = self.base_dataset[idx]
        x0, x1 = self.transform(x)
        return x0, x1

class Linear_cls_Dataset(torch.utils.data.Dataset) :
    def __init__(self, base_dataset, transform) :
        self.transform = transform
        self.base_dataset = base_dataset
        self.len = len(base_dataset)

    def __len__(self) :
        return self.len

    def __getitem__(self, idx) :
        x, target = self.base_dataset[idx]
        x = self.transform(x)
        return x, target

class Random_Rotation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, image) :
        return tf.rotate(image, 90 * np.random.randint(0, 4))

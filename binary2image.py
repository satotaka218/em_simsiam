import argparse
import builtins
from email.policy import default
import math
import os
from statistics import variance
import sys
import random
import shutil
import time
import warnings
from cv2 import transform
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torchvision.datasets as datasets
import torchvision.models as models

import torchvision


def make_cifar_png(image_dataset) :
    for idx in tqdm(range(10)) :
        print("Making image file for index {}".format(idx))
        num_img = 0
        dir_path = './cifar-10-raw/test/' + str(idx)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for image, label in image_dataset:
            if label == idx:
                filename = dir_path +'/' + 'cifar_'+ str(idx) + '_' + str(num_img) + '.png'
                if not os.path.exists(filename):
                    image = image.resize((224, 224))
                    image.save(filename)
                num_img += 1

    print('Success to make CIFAR PNG image files. index={}'.format(idx))

def main():
    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = None)
    make_cifar_png(cifar10_dataset)

if __name__ == '__main__' :
    main()
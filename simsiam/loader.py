# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
from PIL import Image, ImageChops
import random
from matplotlib.pyplot import axis
import numpy as np
from torch import qint32
from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class ThreeCropsTransform:
    def __init__(self, base_transform, third_transform=None):
        self.base_transform = base_transform
        self.third_transform = third_transform if third_transform is not None else base_transform

    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        x0 = self.third_transform(x)
        return [x1, x2, x0]

class MultiCropsTransform:
    '''Inspired by SwAV'''

    def __init__(self, base_transform: list, sizes = [32, 16], scale = (0.08, 1.0)):

        self.base_transform = base_transform
        self.sizes = sizes
        self.scale = scale

    def __call__(self, x) :
        q1 = x.copy()
        k1 = x.copy()
        q2 = x.copy()
        k2 = x.copy()
        for function in self.base_transform :
            for i, size in enumerate(self.sizes):
                if i == 0:
                    if str(type(function)) == "<class 'torchvision.transforms.transforms.RandomResizedCrop'>":
                        function = transforms.RandomResizedCrop(size, scale = self.scale)
                        q1 = function(q1) # 32 x 32
                        k1 = function(k1) # 32 x 32
                    else:
                        q1 = function(q1)
                        k1 = function(k1)
                else:
                    if str(type(function)) == "<class 'torchvision.transforms.transforms.RandomResizedCrop'>":
                        function = transforms.RandomResizedCrop(size, scale = self.scale)
                        q2 = function(q2) #16 x 16
                        k2 = function(k2) # 16 x 16
                    else :
                        q2 = function(q2)
                        k2 = function(k2)
                    
        return [q1, k1, q2, k2]



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class PCA_whitening(object) :
    '''PCA whitening'''

    def __init__(self, epsilon = 1e-4) :
        self.epsilon = epsilon
        self.mean = None 
        self.eigenvalue = None
        self.eigenvector = None 
        self.pca = None 


    def __call__(self, x) :
        x = np.array(x).transpose(2, 0, 1)
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.mean = x.mean(axis = 0, dtype=np.uint8)
        x -= self.mean 
        cov_mat = np.dot(x.T, x) / x.shape[0]
        A, L, _ = np.linalg.svd(cov_mat) # 固有値分解？
        self.PCA_mat = np.dot(np.diag(1. / (np.sqrt(L) + self.epsilon)), A.T)
        x = np.dot(x, self.PCA_mat)
        x = x.reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        pil_image = Image.fromarray(x)

        return pil_image

class Gaussian_Noise(object) :
    '''Gaussian Noise'''

    def __init__(self, sigma = 5):
        self.sigma = sigma 

    def __call__(self, x):
        noise_image = Image.effect_noise((x.height, x.width), self.sigma).convert('RGB') # generate noise image
        synthetic_image = ImageChops.multiply(x, noise_image)

        return synthetic_image

class MyTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        return q


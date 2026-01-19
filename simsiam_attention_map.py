'''
constructing prototypes from simsima latent space by k-means method
'''
import cv2
from cv2 import reduce
from matplotlib import projections
import numpy as np
import scipy.stats as st
import time
import os


from sklearn.cluster import KMeans # import module for k-means from scikit learn
from sklearn.manifold import TSNE # import module for tSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI # import module for ARI score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.resnet as my_ResNet

from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# ==================================================================================================================
#
#   define class for k-means and simsiam pipeline
#
# ==================================================================================================================

class trained_model :
    def __init__(self, checkpoint_path: str, device, ImageNet_flag: bool):
        self.encoder = self._create_model(device, checkpoint_path, ImageNet_flag)
        _target_layer = list(self.encoder.children())[:-3]
        self.feature_extractor = nn.Sequential(*_target_layer)

        # フォルダ自動生成をここに追加
        for d in ["cifar_attention_map"]:
            os.makedirs(d, exist_ok=True)


    # -------------------------------------- #
    #   Model create method
    # -------------------------------------- #

    def _create_model(self, device, checkpoint_path, ImageNet_flag) :
        print('start loding '+ checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        state_dict = checkpoint['state_dict']
        new_state_dict = dict()

        if ImageNet_flag :
            model = models.resnet50(pretrained = False)
            for old_key, value in state_dict.items() :
                new_key = old_key.replace('module.encoder.', '')
                new_state_dict[new_key] = value

        else :
            model = models.resnet18(pretrained = False)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            for old_key, value in state_dict.items() :
                    new_key = old_key.replace('encoder.', '')
                    new_state_dict[new_key] = value
                
            
        msg = model.load_state_dict(new_state_dict, strict = False)
        model.fc = nn.Identity() # delete fc layer
        model.cuda(device)
        model.eval()

        print('end loding '+ checkpoint_path)

        return model

    # -------------------------------------- #
    #   encoding row images to latent space
    # -------------------------------------- #

    def encode_image(self, dataset, transform, class_names, device) :

        with torch.no_grad():
            for n, (image, label) in tenumerate(dataset) :

                normalized_image = transform(image)
                normalized_image = normalized_image.cuda(device, non_blocking = True)

                attention_map = self.feature_extractor(normalized_image.unsqueeze(0))

                attention_map = attention_map.squeeze()
                attention_map = attention_map.cpu().detach().numpy().copy()
                attention_map = attention_map.mean(axis=0)
                attention_map = cv2.normalize(attention_map, None, alpha = 0, beta = 1, norm_type =cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                attention_map = cv2.resize(attention_map, (32, 32), interpolation=cv2.INTER_LINEAR)

                image = np.array(image)

                blend_image = image.copy()
                for i in range(3) :
                    blend_image = blend_image.astype(np.float32)
                    blend_image[:, :, i] *= attention_map
                    blend_image = blend_image.astype(np.uint8)

                fig = plt.figure(figsize=(10, 10))
                plt.subplot(131).imshow(image)
                plt.subplot(132).imshow(attention_map, cmap = 'gray')
                plt.subplot(133).imshow(blend_image)
                plt.savefig(os.path.join(ATT_MAP_DIR, f'{class_names[label]}_{n}.png'))
                plt.close(fig)

                
                if n == 100:
                    break
                


# ==================================================================================================================
#
#  Main method
#
# ==================================================================================================================

# ------------------------------------------------ #
#   setting dicts
# ------------------------------------------------ #

run_tag = input("Run tag (e.g., PhiNet_800 / SimSiam_800 / PhiNet_5): ").strip()
tag = run_tag.lower()
if tag.startswith("simsiam_"):
    run_tag = "SimSiam_" + tag.split("_", 1)[1]
elif tag.startswith("phinet_"):
    run_tag = "PhiNet_" + tag.split("_", 1)[1]
elif tag.startswith("xphinet_"):
    run_tag = "XPhiNet_" + tag.split("_", 1)[1]
run_epochs = int(run_tag.split("_")[-1])

ATT_MAP_DIR = f'./result_figure/{run_tag}/cifar_attention_map'
os.makedirs(ATT_MAP_DIR, exist_ok=True)

normalization_parameter_dict = {'ImageNet': np.array([[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]), 
                                'CIFAR': np.array([[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]])}
checkpoint_path_dict = {
    'ImageNet': './checkpoint/checkpoint_0099.pth.tar',
    'CIFAR10': f'./checkpoint/{run_tag}/checkpoint_{run_epochs:04d}.pth.tar'
}

def main() :

    # ------------------------------------------------ #
    #   hyperparameter setting
    # ------------------------------------------------ #

    checkpoint_path = 'CIFAR10'
    ImageNet_flag = False
    normalization_parameter = normalization_parameter_dict['CIFAR']
    device = 0 if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    # ------------------------------------------------ #
    #   create model
    # ------------------------------------------------ #

    model = trained_model(checkpoint_path_dict[checkpoint_path], device, ImageNet_flag)

    # ------------------------------------------------ #
    #   data loding code
    # ------------------------------------------------ #

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalization_parameter[0], normalization_parameter[1])])
    validate_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = None)
    

    # ------------------------------------------------ #
    #   save image, attention_map, blend_image
    # ------------------------------------------------ #

    model.encode_image(validate_dataset, transform, class_names, device)


if __name__ == '__main__' :
    main()

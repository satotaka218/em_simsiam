'''
constructing prototypes from simsima latent space by k-means method
'''
import cv2
from cv2 import reduce
from matplotlib import projections
import numpy as np
import scipy.stats as st


from sklearn.cluster import KMeans # import module for k-means from scikit learn
from sklearn.manifold import TSNE # import module for tSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI # import module for ARI score

import os
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
# 乱数値を固定
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# ==================================================================================================================
#
#   define class for k-means and simsiam pipeline
#
# ==================================================================================================================

class simsiam_Kmeans :
    def __init__(self, checkpoint_path: str, device, ImageNet_flag: bool, pre_train: bool):
        self.encoder = self._create_model(device, checkpoint_path, ImageNet_flag, pre_train)


    # -------------------------------------- #
    #   Model create method
    # -------------------------------------- #

    def _create_model(self, device, checkpoint_path, ImageNet_flag, pre_train) :
        print('start loding '+ checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        state_dict = checkpoint['state_dict']
        new_state_dict = dict()
        print(len(list(state_dict.keys())))

        if pre_train:

            if ImageNet_flag :
                model = models.resnet50(pretrained = False)
                print(len(list(model.state_dict().keys())))
                for k in list(state_dict.keys()) :
                    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                        new_state_dict[k[len("module.encoder."):]] = state_dict[k]
                print(len(list(new_state_dict.keys())))

            else :
                print('Loading ResNet18')
                model = models.resnet18(pretrained = False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                for old_key, value in state_dict.items() :
                        new_key = old_key.replace('encoder.', '')
                        new_state_dict[new_key] = value

            msg = model.load_state_dict(new_state_dict, strict = False)
            print('end loding '+ checkpoint_path)
        else:
            if ImageNet_flag:
                model = models.resnet50(pretrained=False)
                model.fc = nn.Identity()

            else:
                model = models.resnet18(pretrained=False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()

                
            
        
        model.fc = nn.Identity() # delete fc layer
        model.cuda(device)        

        

        return model

    # -------------------------------------- #
    #   encoding row images to latent space
    # -------------------------------------- #

    def encode_image(self, dataloader, device) :
        
        image_list = []
        label_of_dataset = []
        embeddings = []
        self.encoder.eval()

        with torch.no_grad():
            for i, (images, labels) in tenumerate(dataloader) :
                images = images.cuda(device)
                outputs = self.encoder(images)
                outputs = outputs.cpu().detach().numpy().copy()
                [image_list.append(image.cpu().detach().numpy().copy().tolist()) for image in images]
                [label_of_dataset.append(int(label.cpu().detach())) for label in labels]
                for i in range(outputs.shape[0]) :
                    embeddings.append(outputs[i, :].tolist())


        return image_list, np.array(label_of_dataset), np.array(embeddings)


    # ------------------------------------------------ #
    #   k-means clustering with simsiam embeddings
    # ------------------------------------------------ #

    @staticmethod
    def kmeans(data, num_cluster = 10, init = 'random', num_iterations = 1000, num_initialization = 30, random_state = 0) :

        kmeans = KMeans(n_clusters = num_cluster, init = init, max_iter = num_iterations, tol=1e-5,  n_init = num_initialization, random_state = random_state)
        result = kmeans.fit(data)
        print('SSE of Intra-cluster: ' + format(kmeans.inertia_, '.2f'))
        distance_per_embedding = kmeans.fit_transform(data)

        prototypes = (kmeans.cluster_centers_.copy() - kmeans.cluster_centers_.mean(axis = 0)) / np.std(kmeans.cluster_centers_, axis=0)

        return prototypes, result.labels_, distance_per_embedding


# ==================================================================================================================
#
#   define visualize top of k result of k-means clustering
#
# ==================================================================================================================

def plot_tile(top_of_k_image, top_of_k_label, class_names, normalize_parameters_list: np.ndarray, pm = 10) :
    fig, ax = plt.subplots(pm, pm, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # ------------------------------------------------ #
    #   rehsape for broadcast
    # ------------------------------------------------ #
    normalization_mean = normalize_parameters_list[0].reshape(3, 1, 1)
    normalization_std = normalize_parameters_list[1].reshape(3, 1, 1)

    for i in range(pm):
        for j in range(pm):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            img = (top_of_k_image[i][j].copy() + normalization_mean) * normalization_std
            # img = cv2.resize(img.transpose(1, 2, 0), (40, 40))
            img = _adjust(img.transpose(1, 2, 0))
            # ax[i, j].imshow(img.transpose(1, 2, 0), cmap="bone")
            ax[i, j].imshow(img, cmap="bone")
            ax[i, j].set_title("{}".format(class_names[top_of_k_label[i][j]]), x = 0.5, y = -0.4, fontsize=12, color = "green")
    plt.savefig(f'{run_dir}/top_of_10.png')
    plt.show()

# ==================================================================================================================
#
#   輝度値を調整するメソッド
#
# ==================================================================================================================

def _adjust(img, alpha=1.4, beta=0.0):
    # 積和演算を行う。
    dst = alpha * img + beta

    return dst

# ==================================================================================================================
#
#   Method to get k samples which closest to the prototype
#
# ==================================================================================================================

def top_of_k(image_list, label_of_kmeans, label_of_dataset, distance_per_embedding, prototypes, k = 10):
    '''可読性が低すぎるので改善が必要'''
    top_of_k_image = []
    top_of_k_label = []
    for cluster_label in range(len(prototypes)):
        images_per_cluster = [image_list[i] for i in np.where(label_of_kmeans == cluster_label)[0]]
        labels_per_cluster = [label_of_dataset[i] for i in np.where(label_of_kmeans == cluster_label)[0]]
        distances_per_cluster =np.array([distance_per_embedding[i] for i in np.where(label_of_kmeans== cluster_label)[0]])
        argsort_distance = np.argsort(distances_per_cluster)[0]
        top_k_images_per_cluster = [images_per_cluster[top_k_index] for top_k_index in argsort_distance[:k]]
        top_k_label_per_cluster = [labels_per_cluster[top_k_index] for top_k_index in argsort_distance[:k]]
        top_of_k_image.append(top_k_images_per_cluster)
        top_of_k_label.append(top_k_label_per_cluster)

    return top_of_k_image, top_of_k_label


# ==================================================================================================================
#
#  Method to compress dimension
#
# ==================================================================================================================

def compress_tSNE(embeddings: np.ndarray, prototypes: np.ndarray, labels_of_dataset: np.ndarray, labels_of_kmeans: np.ndarray, dimension = 2, perplexity = 5, random_state = 0):
    embeddings_and_prototypes = np.concatenate([embeddings, prototypes]) # last 10 components are prototype
    '''vector normalize'''
    # norm_each_row = np.linalg.norm(embeddings_and_prototypes.copy(), axis=1, keepdims=True)
    # embeddings_and_prototypes = embeddings_and_prototypes.copy() / norm_each_row
    # tsne = TSNE(n_components=dimension, perplexity=perplexity, random_state=random_state)
    # reduced = tsne.fit_transform(embeddings_and_prototypes)

    tsne = TSNE(n_components=dimension, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(embeddings)

    

    class_label_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # visualize
    if dimension == 2 :

        reduced_embeddings = reduced[:10000]
        reduced_prototypes = reduced[-10:]

        # 軸あり
        fig = plt.figure()
        for i in range(10) :
            embeddings_per_class = reduced_embeddings[labels_of_dataset == i]
            plt.scatter(embeddings_per_class[:, 0], embeddings_per_class[:, 1], label = class_label_names[i], s = 5)

        # plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c = labels_of_dataset, cmap = 'tab10', s = 5, alpha = 0.5) # visualize embbeded data
        # for i in range(10) :
        #     marker = "$" + str(i) + "$"
        #     plt.scatter(reduced_prototypes[i, 0], reduced_prototypes[i, 1], c = 'k', s = 50, marker= marker)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{run_dir}/tsne_CIFAR10_100_ResNet18.png')
        plt.show()

    else :
        
        '''vector normalize'''
        norm_each_row = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced.copy() / norm_each_row
        reduced_embeddings = reduced[:10000]
        reduced_prototypes = reduced[-10:]

        theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
        THETA, PHI = np.meshgrid(theta, phi)
        X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

        '''scatter in 3D'''
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")

        # ax = Axes3D(fig)
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c = labels_of_dataset, cmap = 'tab10', s = 5, alpha = 0.5)
        plt.savefig(f'{run_dir}/CIFAR10_ResNet18_3D.png')
        plt.show()

# ==================================================================================================================
#
#  Method for principal component analisys
#
# ==================================================================================================================

def compress_PCA(embeddings: np.ndarray, labels_of_dataset: np.ndarray, dimension = 2) :
    
    pca = PCA()
    pca.fit(embeddings)
    pca_result = pca.transform(embeddings)

    class_label_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # ------------------------------------------------ #
    #   visualization code
    # ------------------------------------------------ #
    if dimension == 2:
        
        '''Cumulative contribution rate'''
        contribution_ratios = pca.explained_variance_ratio_
        cumulative_contribution_ratios = contribution_ratios.cumsum() # Cumulative sum
        plt.figure(figsize = (10, 10))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.plot([i for i in range(contribution_ratios.shape[0])], cumulative_contribution_ratios)
        plt.title('Cumulative Contribution Rate')
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.bar([i for i in range(contribution_ratios.shape[0])], contribution_ratios)
        plt.title('Contiribution Rate of each Principal Component')
        plt.grid()
        plt.show()

        '''scatter in 2D'''
        plt.figure()
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        for i in range(10) :
            embeddings_per_class = pca_result[labels_of_dataset == i]
            plt.scatter(embeddings_per_class[:, 0], embeddings_per_class[:, 1] , label = class_label_names[i], cmap = 'tab10', s = 5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{run_dir}/PCA_CIFAR10_ResNet18.png')
        plt.show()

        print('第一主成分の寄与率: {}\n第二主成分の寄与率: {}\n第三種成分の寄与率: {}'.format(contribution_ratios[0], contribution_ratios[1], contribution_ratios[2]))
        print('第二主成分までの累積寄与率: {}\n第三主成分までの累積寄与率: {}'.format(cumulative_contribution_ratios[0], cumulative_contribution_ratios[1]))
    else :

        '''vector normalize'''
        norm_each_row = np.linalg.norm(pca_result, axis=1, keepdims=True)
        pca_result = pca_result.copy() / norm_each_row

        '''Cumulative contribution rate'''
        contribution_ratios = pca.explained_variance_ratio_

        theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
        THETA, PHI = np.meshgrid(theta, phi)
        X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

        '''scatter in 3D'''
        fig = plt.figure(figsize=(10, 10))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax = fig.gca(projection = '3d')
        # ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")

        # ax = Axes3D(fig)
        for i in range(10) :
            embeddings_per_class = pca_result[labels_of_dataset == i]
            ax.scatter(embeddings_per_class[:, 0], embeddings_per_class[:, 1], embeddings_per_class[:, 2], label = class_label_names[i], s = 5, alpha = 0.5)
    
        plt.show()

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

run_dir = f'./result_figure/{run_tag}'
os.makedirs(run_dir, exist_ok=True)

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
    plot_dimension = 2
    pretrain = True

    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    # ------------------------------------------------ #
    #   create model
    # ------------------------------------------------ #

    model = simsiam_Kmeans(checkpoint_path_dict[checkpoint_path], device, ImageNet_flag, pretrain)

    # ------------------------------------------------ #
    #   data loding code
    # ------------------------------------------------ #

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalization_parameter[0], normalization_parameter[1])])
    validate_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size = 16, shuffle = False, num_workers=4, pin_memory=True, drop_last = True)

    # ------------------------------------------------ #
    #   encode images
    # ------------------------------------------------ #

    image_list, label_of_dataset, embeddings = model.encode_image(validate_loader, device)
    '''vector normalize'''
    norm_each_row = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.copy() / norm_each_row

    # ------------------------------------------------ #
    #   k-means clustering
    # ------------------------------------------------ #

    prototypes, label_of_kmeans, distance_per_embedding = simsiam_Kmeans.kmeans(embeddings)
    print(prototypes)

    # ------------------------------------------------------------------ #
    #   Compress high dimension to 2d space and visualize tSNE result
    # ------------------------------------------------------------------ #

    compress_tSNE(embeddings, prototypes, label_of_dataset, label_of_kmeans, dimension=plot_dimension)

    # --------------------------------#
    #   pricipal component analisys
    # ------------------------------- #

    compress_PCA(embeddings, label_of_dataset, dimension=plot_dimension)

    # ------------------------------------------------ #
    #   get top of k samples and plot result
    # ------------------------------------------------ #

    top_of_k_image, top_of_k_label = top_of_k(image_list, label_of_kmeans, label_of_dataset, distance_per_embedding, prototypes, k = 10)
    plot_tile(top_of_k_image, top_of_k_label, class_names, normalization_parameter)

    # ------------------------------------------------ #
    #   calculate adjust rand index score
    # ------------------------------------------------ #
    ari_score = ARI(np.array(label_of_dataset), label_of_kmeans)
    print('adjust rand index score: {}'.format(ari_score))


if __name__ == '__main__' :
    main()

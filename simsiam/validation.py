import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms, datasets
import numpy as np


class KNNValidation(object):
    """
    kNN evaluation for representation quality.

    Notes
    - This class extracts backbone features (encoder without the last module).
    - Similarity is computed by dot product (cosine after L2-normalization).
    - Prediction is performed by majority vote among top-K nearest neighbors.
      (K=1 reduces to standard 1-NN.)
    """

    def __init__(
        self,
        data_root,
        batch_size,
        num_workers,
        model,
        dim=512,
        K=200,
        num_classes=10,
        dataset_name="cifar10",
        input_size=32,
        normalization_parameter=None,
    ):
        # model is expected to be `model.encoder` (a ResNet with projector at `.fc`).
        encoder_without_last = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*encoder_without_last)

        self.dim = int(dim)
        self.K = int(K)
        self.num_classes = int(num_classes)

        # device follows feature_extractor params
        try:
            self.device = next(self.feature_extractor.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if normalization_parameter is None:
            if dataset_name == "stl10":
                normalization_parameter = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            else:
                normalization_parameter = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        transform_ops = []
        if dataset_name == "stl10":
            transform_ops.extend([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
            ])

        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalization_parameter[0], normalization_parameter[1]),
        ])
        base_transforms = transforms.Compose(transform_ops)

        if dataset_name == "stl10":
            train_dataset = datasets.STL10(
                root=data_root,
                split="train",
                download=True,
                transform=base_transforms,
            )
            val_dataset = datasets.STL10(
                root=data_root,
                split="test",
                download=True,
                transform=base_transforms,
            )
        else:
            train_dataset = datasets.CIFAR10(
                root=data_root,
                train=True,
                download=True,
                transform=base_transforms,
            )
            val_dataset = datasets.CIFAR10(
                root=data_root,
                train=False,
                download=True,
                transform=base_transforms,
            )

        # drop_last=False : use all samples for eval
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @torch.no_grad()
    def _extract_train_features(self):
        """Return (train_features: [N, D], train_labels: [N]) on self.device."""
        self.feature_extractor.eval()

        n_data = len(self.train_dataloader.dataset)
        feat_dim = self.dim

        train_features = torch.zeros((n_data, feat_dim), device=self.device, dtype=torch.float32)
        if hasattr(self.train_dataloader.dataset, "targets"):
            train_label_list = self.train_dataloader.dataset.targets
        else:
            train_label_list = self.train_dataloader.dataset.labels
        train_labels = torch.tensor(train_label_list, device=self.device, dtype=torch.long)

        start = 0
        for inputs, _ in self.train_dataloader:
            inputs = inputs.to(self.device, non_blocking=True)

            feats = self.feature_extractor(inputs)
            feats = torch.squeeze(feats)  # [B, D] (avgpool output)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)

            feats = nn.functional.normalize(feats, dim=1)  # L2 normalize
            bs = feats.size(0)
            train_features[start:start + bs] = feats
            start += bs

        train_features = train_features[:start]
        train_labels = train_labels[:start]
        return train_features, train_labels

    @torch.no_grad()
    def _topk_retrieval(self):
        """
        Extract features from validation split and retrieve top-K neighbors from train features.
        Majority vote among neighbors is used for top-1 prediction.
        """
        train_features, train_labels = self._extract_train_features()  # [N, D], [N]

        total = 0
        correct = 0

        for inputs, targets in self.val_dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            feats = self.feature_extractor(inputs)
            feats = torch.squeeze(feats)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)

            feats = nn.functional.normalize(feats, dim=1)

            # cosine similarity via dot product after normalization: [B, N]
            sim = torch.mm(feats, train_features.t())

            k = min(self.K, sim.size(1))
            _, nn_idx = sim.topk(k, dim=1, largest=True, sorted=True)  # [B, k]
            nn_labels = train_labels[nn_idx]  # [B, k]

            # majority vote
            counts = torch.zeros((nn_labels.size(0), self.num_classes), device=self.device, dtype=torch.long)
            ones = torch.ones_like(nn_labels, dtype=torch.long)
            counts.scatter_add_(dim=1, index=nn_labels, src=ones)
            pred = counts.argmax(dim=1)

            total += targets.size(0)
            correct += (pred == targets).sum().item()

        return correct / total

    def eval(self):
        return self._topk_retrieval()

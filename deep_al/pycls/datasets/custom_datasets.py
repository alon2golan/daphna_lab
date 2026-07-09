import torch
import torchvision
from PIL import Image
import numpy as np
import deep_al.pycls.datasets.utils as ds_utils
from torchvision.datasets import VisionDataset


def _load_scale_preserving_normalized(ds_name, train, features_tag):
    """
    L2-normalize per sample, then rescale by the TRAIN split's mean raw norm.
    Removes per-sample norm variance (what hurts distance-based methods and
    biases the linear head) WITHOUT shrinking the global input scale — the
    OPTIM hyperparameters (BASE_LR/MAX_EPOCH) were tuned on raw-scale features
    and plain unit-normalization underfits badly at low budget (train acc
    ~19% vs ~95% @100 labels on CIFAR100). The scale constant always comes
    from the train split so train/test share one transform.
    """
    feats = ds_utils.load_features(ds_name, train=train, normalized=False, features_tag=features_tag)
    train_feats = feats if train else ds_utils.load_features(ds_name, train=True, normalized=False, features_tag=features_tag)
    scale = np.linalg.norm(train_feats, axis=1).mean()
    return feats / np.linalg.norm(feats, axis=1, keepdims=True) * scale


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True, only_features=False, features_tag=None, normalize_features=False):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = _load_scale_preserving_normalized("CIFAR10", train, features_tag) if normalize_features \
            else ds_utils.load_features("CIFAR10", train=train, normalized=False, features_tag=features_tag)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)


        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True, only_features=False, features_tag=None, normalize_features=False):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = _load_scale_preserving_normalized("CIFAR100", train, features_tag) if normalize_features \
            else ds_utils.load_features("CIFAR100", train=train, normalized=False, features_tag=features_tag)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(STL10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.targets = self.labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SCENARIO_A(VisionDataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__(root="", transform=transform, target_transform=target_transform)
        self.features = X  # shape: (N, 2)
        self.targets = y     # labels
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]   # tensor shape: (2,)
        t = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            t = self.target_transform(t)
        return x, t

class HALF_MOON(VisionDataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__(root="", transform=transform, target_transform=target_transform)
        self.features = X  # shape: (N, 2)
        self.targets = y     # labels
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]   # tensor shape: (2,)
        t = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            t = self.target_transform(t)
        return x, t

class CIFAR10_SPARSE(VisionDataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__(root="", transform=transform, target_transform=target_transform)
        self.features = X  # shape: (N, 2)
        self.targets = y     # labels
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]   # tensor shape: (2,)
        t = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            t = self.target_transform(t)
        return x, t
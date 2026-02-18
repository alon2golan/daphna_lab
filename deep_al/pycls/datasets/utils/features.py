import numpy as np
import torch

# DATASET_FEATURES_DICT = {
#     'train':
#         {
#             'CIFAR10':'/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/pretext/features_seed{seed}.npy',
#             'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed{seed}.npy',
#             'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed{seed}.npy',
#             'IMAGENET50': '../../dino/runs/trainfeat.pth',
#             'IMAGENET100': '../../dino/runs/trainfeat.pth',
#             'IMAGENET200': '../../dino/runs/trainfeat.pth',
#         },
#     'test':
#         {
#             'CIFAR10': '/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/pretext/features_seed{seed}.npy',
#             'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed{seed}.npy',
#             'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
#             'IMAGENET50': '../../dino/runs/testfeat.pth',
#             'IMAGENET100': '../../dino/runs/testfeat.pth',
#             'IMAGENET200': '../../dino/runs/testfeat.pth',
#         }
# }

DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'/cs/labs/daphna/itai.david/representations_bank/cifar-10_simclr/pretext/features_seed1.npy',
            'CIFAR100': '/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy',
            'CIFAR100_TypiclustRepo': '/cs/labs/daphna/alongol/data/embeddings/cifar-100/downloaded_from_typiclust/features_seed1.npy',
            'CIFAR100_alon': '/cs/labs/daphna/alongol/data/embeddings/cifar-100/alon_20260217/features_seed1.npy',
            'TINYIMAGENET': '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/features_seed1.npy',
            'IMAGENET50': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET100': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET200': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'IMAGENET': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/train_features.npy',
            'SCENARIO_A': f"/cs/labs/daphna/itai.david/py_repos/points_A.txt",
            'HALF_MOON': f"/cs/labs/daphna/itai.david/py_repos/points_HALF_MOON_2.txt",
        },
    'test':
        {
            'CIFAR10': '/cs/labs/daphna/itai.david/representations_bank/cifar-10_simclr/pretext/test_features_seed1.npy',
            'CIFAR100': '/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy',
            'CIFAR100_TypiclustRepo  ': '/cs/labs/daphna/alongol/data/embeddings/cifar-100/downloaded_from_typiclust/features_seed1.npy',
            'CIFAR100_alon': '/cs/labs/daphna/alongol/data/embeddings/cifar-100/alon_20260217/test_features_seed1.npy',
            'TINYIMAGENET': '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy',
            'IMAGENET50': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET100': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET200': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'SCENARIO_A': f"/cs/labs/daphna/itai.david/py_repos/TEST_points_A.txt",
            'HALF_MOON': f"/cs/labs/daphna/itai.david/py_repos/TEST_points_HALF_MOON_2.txt",
        }
}

def get_dataset_name_with_tag(ds_name, features_tag=None):
    """
    Constructs the dataset name with the features tag if provided.
    Allows specifying different feature representations for any dataset (e.g., 'itai', 'alon').
    
    Args:
        ds_name: Base dataset name (e.g., 'CIFAR100', 'CIFAR10', 'TINYIMAGENET')
        features_tag: Optional tag to specify which features to use (e.g., 'itai', 'alon')
    
    Returns:
        Dataset name with tag if applicable (e.g., 'CIFAR100_alon', 'CIFAR10_custom')
    """
    if features_tag:
        return f'{ds_name}_{features_tag}'
    return ds_name

def load_features(ds_name, seed=1, train=True, normalized=True, features_tag=None):
    """
    Load pretrained features for a dataset.
    
    Args:
        ds_name: Dataset name
        seed: Random seed
        train: Whether to load training or test features
        normalized: Whether to normalize features
        features_tag: Optional tag to specify which features to use for CIFAR100
    
    Returns:
        Loaded features as numpy array
    """
    # Construct the full dataset name with tag if provided
    full_ds_name = get_dataset_name_with_tag(ds_name, features_tag)
    
    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][full_ds_name].format(seed=seed)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    elif fname.endswith('.txt'):
        normalized = False
        features = np.loadtxt(fname)[:, :2].astype(np.float32)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features
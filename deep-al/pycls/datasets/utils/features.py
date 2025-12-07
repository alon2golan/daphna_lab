import numpy as np
import torch
import os
from sklearn.datasets import make_blobs

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
            'TINYIMAGENET': '/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy',
            'IMAGENET50': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET100': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET200': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'IMAGENET': '/cs/labs/daphna/itai.david/representations_bank/imagenet_dinov2/val_features.npy',
            'SCENARIO_A': f"/cs/labs/daphna/itai.david/py_repos/TEST_points_A.txt",
            'HALF_MOON': f"/cs/labs/daphna/itai.david/py_repos/TEST_points_HALF_MOON_2.txt",
        }
}

def load_features(ds_name, seed=1, train=True, normalized=True):
    " load pretrained features for a dataset "
    
    # Handle SCENARIO_A generation dynamically
    if ds_name == 'SCENARIO_A':
        # Load from the centralized file
        # Assuming the file is at /cs/labs/daphna/dvir_mateless/lab_repo/data/scenario_a_data.npy
        # We need to find the path relative to this file or absolute
        # Going up from deep-al/pycls/datasets/utils/features.py to lab_repo/data
        
        # Hardcoded path for now based on workspace structure, or relative path
        # Current file: .../deep-al/pycls/datasets/utils/features.py
        # Data file: .../lab_repo/data/scenario_a_data.npy
        # That is ../../../../../data/scenario_a_data.npy relative to this file
        
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../data/scenario_a_data.npy'))
        
        if os.path.exists(base_path):
            data = np.load(base_path, allow_pickle=True).item()
            features = data['points']
            # If train=False (test set), we might need to handle that. 
            # But currently generate_scenario_a.py only generates 1000 points (train).
            # If test is needed, we should generate it too.
            # For now, assuming train=True or using the same points.
            if not train:
                # Fallback or generate test set on the fly if not in file
                # Or better, update generate_scenario_a.py to save test set too.
                # For now, let's just return the first 250 points if test is requested from this file
                features = features[:250] 
        else:
            # Fallback to generation if file missing
            print(f"Data file for SCENARIO_A not found at {base_path}. Generating data on the fly.")
            N = 1000 if train else 250
            centers = [[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [0.0, 0.0]]
            features, _ = make_blobs(n_samples=N, centers=centers, n_features=2, cluster_std=1.0, random_state=seed)
            features = features.astype(np.float32)

        if normalized:
             features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features

    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)
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
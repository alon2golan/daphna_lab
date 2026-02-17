import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import models
from copy import copy
# add path to TypiClust/deep-al
import sys
sys.path.append(os.path.abspath('/cs/labs/daphna/tomer_yaacoby/pythonProjects/oracle-teacher-proj/TypiClust/deep-al'))  # TODO change properly
import pycls.core.builders as model_builder
import pycls.core.optimizer as optim

from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min



import argparse


# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------
def get_logger(name):
    """Retrieves the logger."""
    logger = logging.getLogger(name)
    return logger

logger = get_logger(__name__)


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _cluster_labels(x_np: np.ndarray,
                    num_classes: int,
                    method: str = "kmeans",
                    seed: int = 42,
                    *,
                    # DBSCAN-specific (optional)
                    dbscan_eps: float | None = None,
                    dbscan_min_samples: int = 10,
                    dbscan_metric: str = "euclidean",
                    dbscan_use_pca: bool = True,
                    pca_dim: int = 128) -> np.ndarray:
    """
    Return integer cluster labels for x_np using the requested method.
    Adds 'dbscan' with:
      - optional auto-eps to target num_classes
      - noise (-1) reassignment
      - fallback merge/split to get exactly num_classes labels
    """
    method = method.lower()

    # ---------- existing 'kmeans' / 'gmm' / 'agglomerative' branches above ----------

    if method == "dbscan":
        # 1) scale + (optional) PCA for speed/stability
        X = StandardScaler().fit_transform(x_np).astype("float32")
        if dbscan_use_pca and X.shape[1] > pca_dim:
            X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

        N = X.shape[0]

        def run_dbscan(eps: float) -> np.ndarray:
            return DBSCAN(
                eps=eps,
                min_samples=dbscan_min_samples,
                metric=dbscan_metric,
                n_jobs=-1  # use all cores if available
            ).fit_predict(X)

        def count_clusters(lbls: np.ndarray) -> int:
            uniq = np.unique(lbls)
            return int(np.sum(uniq != -1))  # exclude noise

        # 2) If eps not provided, auto-pick via k-dist percentiles and adjust
        if dbscan_eps is None:
            # Heuristic: start near the 95th percentile of k-distances (k = min_samples)
            k = min(dbscan_min_samples, max(2, int(0.01 * N)))  # safe bounds
            nn = NearestNeighbors(n_neighbors=k, metric=dbscan_metric, n_jobs=-1).fit(X)
            dists, _ = nn.kneighbors(X)
            kth = dists[:, -1]  # k-th NN distance per point
            low_q, hi_q = 50.0, 99.5
            q = 95.0
            eps = float(np.percentile(kth, q))

            # adjust q to push cluster count toward target
            target = num_classes
            max_iter = 12
            for _ in range(max_iter):
                labels = run_dbscan(eps)
                k_found = count_clusters(labels)
                if k_found == target:
                    break
                # monotonic-ish: larger eps → fewer clusters (and less noise)
                if k_found > target:
                    # too many clusters → increase eps
                    low_q = q
                    q = (q + hi_q) / 2
                else:
                    # too few clusters → decrease eps
                    hi_q = q
                    q = (q + low_q) / 2
                eps = float(np.percentile(kth, q))
            dbscan_eps = eps  # keep chosen value for logging/debug
            labels = labels  # from the last run
        else:
            labels = run_dbscan(dbscan_eps)

        # 3) Reassign noise (-1) to nearest cluster centroid if any clusters exist
        lbls = labels.copy()
        uniq = np.unique(lbls)
        cluster_ids = [c for c in uniq if c != -1]

        if len(cluster_ids) == 0:
            # DBSCAN found no clusters → fall back to MiniBatchKMeans
            mbk = MiniBatchKMeans(n_clusters=num_classes, random_state=seed, n_init="auto")
            return mbk.fit_predict(X).astype(np.int64)

        # centroids of found clusters
        centroids = np.vstack([X[lbls == c].mean(axis=0) for c in cluster_ids])
        # assign noise to nearest centroid
        noise_mask = (lbls == -1)
        if np.any(noise_mask):
            assign, _ = pairwise_distances_argmin_min(X[noise_mask], centroids, metric="euclidean")
            lbls[noise_mask] = np.array(cluster_ids, dtype=int)[assign]

        # 4) Ensure exactly num_classes labels:
        #    - if too many clusters → merge via KMeans on centroids
        #    - if too few          → split the largest clusters via KMeans
        def relabel_to_0_k_minus_1(lbls_in: np.ndarray) -> np.ndarray:
            # map arbitrary labels to 0..K-1 (stable order by size)
            counts = [(c, np.sum(lbls_in == c)) for c in np.unique(lbls_in)]
            counts.sort(key=lambda t: t[1], reverse=True)
            mapping = {c: i for i, (c, _) in enumerate(counts)}
            return np.vectorize(mapping.get)(lbls_in)

        # recompute after noise reassignment
        cluster_ids = np.unique(lbls)
        K_found = len(cluster_ids)

        if K_found == num_classes:
            return relabel_to_0_k_minus_1(lbls).astype(np.int64)

        # Build per-cluster centroids for merging/splitting
        centroids = np.vstack([X[lbls == c].mean(axis=0) for c in cluster_ids])

        if K_found > num_classes:
            # Merge clusters down to num_classes using KMeans on centroids,
            # then propagate the centroid labels back to points.
            km = MiniBatchKMeans(n_clusters=num_classes, random_state=seed, n_init="auto")
            centroid_labels = km.fit_predict(centroids)
            # map each original cluster id to a merged id
            id_to_new = {cid: centroid_labels[i] for i, cid in enumerate(cluster_ids)}
            merged = np.vectorize(id_to_new.get)(lbls)
            return relabel_to_0_k_minus_1(merged).astype(np.int64)

        else:  # K_found < num_classes → split largest clusters
            # We’ll run KMeans on the full data with init = current centroids,
            # asking for num_classes final clusters.
            # (If you want stricter “split only largest”, implement iterative splits.)
            km = MiniBatchKMeans(
                n_clusters=num_classes,
                random_state=seed,
                n_init=1,
                init=centroids if centroids.shape[0] >= num_classes else "k-means++"
            )
            return relabel_to_0_k_minus_1(km.fit_predict(X)).astype(np.int64)

    # ---------- default fallback ----------
    # (your existing 'kmeans' as the final else)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_classes, random_state=seed, n_init="auto")
    return km.fit_predict(x_np).astype(np.int64)

# def _cluster_labels(x_np: np.ndarray,
#                     num_classes: int,
#                     method: str = "kmeans",
#                     seed: int = 42) -> np.ndarray:
#     """
#     Return integer cluster labels for x_np using the requested method.
#     Methods: 'kmeans', 'gmm', 'agglomerative' (ward).
#     """
#     method = method.lower()
#     if method == "kmeans":
#         km = KMeans(n_clusters=num_classes, random_state=seed, n_init="auto")
#         return km.fit_predict(x_np).astype(np.int64)
#
#     if method == "gmm":
#         # 1) scale + cast to float32
#         X = StandardScaler(with_mean=True, with_std=True).fit_transform(x_np).astype(np.float32)
#
#         # 2) PCA to a compact space (tune n_components if needed)
#         d_target = 128 if X.shape[1] > 128 else X.shape[1]
#         if d_target < X.shape[1]:
#             X = PCA(n_components=d_target, random_state=seed, svd_solver="auto").fit_transform(X)
#
#         # 3) Fit a faster GMM
#         gm = GaussianMixture(
#             n_components=num_classes,
#             covariance_type="diag",      # much faster than "full"
#             max_iter=50,                 # fewer EM iterations
#             tol=1e-3,
#             init_params="kmeans",        # good init
#             n_init=1,                    # fewer restarts
#             reg_covar=1e-6,
#             random_state=seed,
#             verbose=0
#         )
#
#         # Optional: subsample to fit, then predict on all
#         # idx = np.random.default_rng(seed).choice(len(X), size=min(20000, len(X)), replace=False)
#         # gm.fit(X[idx])
#         # return gm.predict(X).astype(np.int64)
#
#         gm.fit(X)
#         return gm.predict(X).astype(np.int64)
#
#     if method == "agglomerative":
#         # Ward linkage requires Euclidean metric and works best on scaled features.
#         # (Assumes your features are already reasonable; scale if needed.)
#         X = StandardScaler().fit_transform(x_np).astype(np.float32)
#         ag = AgglomerativeClustering(n_clusters=num_classes, linkage="ward")
#         return ag.fit_predict(X).astype(np.int64)
#
#     raise ValueError(f"Unknown cluster method '{method}'")



def _build_model(img_shape: int,
                 num_classes: int,
                 model_name: str = "resnet18",
                 representation_model='simclr') -> nn.Module:
    if model_name.lower() == "resnet18" or model_name.lower() == "resnet50":
        model = model_builder.build_model(cfg)

        # model = models.resnet18(weights=None)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name in ('vit', 'vit_b_16'):
        # torchvision ViT supports an `image_size` kwarg
        _, H, W = img_shape
        # In recent torchvision versions you can do:
        model = models.vit_b_16(weights=None, image_size=H)
        # Replace the default head with one matching num_classes
        in_feat = model.heads.head.in_features
        model.heads = nn.Linear(in_feat, num_classes)

    elif model_name == 'linear':
        # your custom FeaturesNet; assumes cfg.DATASET.REPRESENTATION_DIM is defined
        if representation_model == 'resnet50':
            num_features = 2048
        elif representation_model == 'vit':
            num_features = 768
        else:
            num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100','IMAGENET200'] else 512
        model = FeaturesNet(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model type '{model_name}'")
    return model


def _train(model: nn.Module,
           loader: DataLoader,
           device: torch.device,
           epochs: int = 5,
           lr: float = 1e-3,
           cfg: str = None) -> None:
    model.to(device)
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # Construct the optimizer
    opt = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(opt.state_dict())
    model_init_state = deepcopy(model.state_dict().copy())
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            train_loss += loss.item()
            opt.step()
        if epoch == 0 or (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}", flush=True)


@torch.no_grad()
def _accuracy(model: nn.Module,
              loader: DataLoader,
              device: torch.device) -> float:
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ---------- Asymmetric-noise helpers ----------

# CIFAR-10 (torchvision order):
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
CIFAR10_MAPPING = {
    0: 2, 2: 0,   # airplane <-> bird
    1: 9, 9: 1,   # automobile <-> truck
    3: 5, 5: 3,   # cat <-> dog
    4: 7, 7: 4,   # deer <-> horse
    6: 8, 8: 6,   # frog <-> ship
}

CIFAR100_SUPERCLASSES = {
    "aquatic mammals": ["beaver","dolphin","otter","seal","whale"],
    "fish": ["aquarium_fish","flatfish","ray","shark","trout"],
    "flowers": ["orchid","poppy","rose","sunflower","tulip"],
    "food containers": ["bottle","bowl","can","cup","plate"],
    "fruit and vegetables": ["apple","mushroom","orange","pear","sweet_pepper"],
    "household electrical devices": ["clock","keyboard","lamp","telephone","television"],
    "household furniture": ["bed","chair","couch","table","wardrobe"],
    "insects": ["bee","beetle","butterfly","caterpillar","cockroach"],
    "large carnivores": ["bear","leopard","lion","tiger","wolf"],
    "large man-made outdoor things": ["bridge","castle","house","road","skyscraper"],
    "large natural outdoor scenes": ["cloud","forest","mountain","plain","sea"],
    "large omnivores and herbivores": ["camel","cattle","chimpanzee","elephant","kangaroo"],
    "medium-sized mammals": ["fox","porcupine","possum","raccoon","skunk"],
    "non-insect invertebrates": ["crab","lobster","snail","spider","worm"],
    "people": ["baby","boy","girl","man","woman"],
    "reptiles": ["crocodile","dinosaur","lizard","snake","turtle"],
    "small mammals": ["hamster","mouse","rabbit","shrew","squirrel"],
    "trees": ["maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"],
    "vehicles 1": ["bicycle","bus","motorcycle","pickup_truck","train"],
    "vehicles 2": ["lawn_mower","rocket","streetcar","tank","tractor"],
}

def _build_cycle_mapping(K: int):
    """Generic fallback mapping c -> (c+1)%K."""
    return {c: (c + 1) % K for c in range(K)}

def _maybe_get_class_to_idx(dataset):
    """
    Try to recover a class_to_idx mapping from the dataset your Data(cfg) returns.
    Works for torchvision-style datasets; otherwise returns None.
    """
    if hasattr(dataset, "class_to_idx") and isinstance(dataset.class_to_idx, dict):
        return dataset.class_to_idx
    if hasattr(dataset, "classes") and isinstance(dataset.classes, (list, tuple)):
        return {name: i for i, name in enumerate(dataset.classes)}
    # Some wrappers store name lists in dataset.base or dataset.dataset
    for attr in ["base", "dataset", "data", "ds"]:
        obj = getattr(dataset, attr, None)
        if obj is not None:
            c2i = _maybe_get_class_to_idx(obj)
            if c2i is not None:
                return c2i
    return None

def _build_cifar100_mapping_from_c2i(class_to_idx: dict):
    """Create a 5-cycle within each CIFAR-100 superclass."""
    mapping = {}
    for group in CIFAR100_SUPERCLASSES.values():
        idxs = [class_to_idx[name] for name in group]
        for i, src in enumerate(idxs):
            dst = idxs[(i + 1) % len(idxs)]
            mapping[src] = dst
    assert len(mapping) == 100, f"Expected 100 entries, got {len(mapping)}"
    return mapping

def _make_asym_mapping(dataset_name: str, num_classes: int, dataset, labels_are_clusters: bool):
    """
    Decide which asymmetric mapping to use, based on dataset_name and label type.
    If labels are KMeans cluster ids (0..K-1), we cannot use names, so use a cycle.
    """
    name = dataset_name.lower()
    if labels_are_clusters:
        # Cluster ids have no semantic meaning → safe cycle
        return _build_cycle_mapping(num_classes)

    if name == "cifar10":
        return CIFAR10_MAPPING

    if name == "cifar100":
        c2i = _maybe_get_class_to_idx(dataset)
        if c2i is not None:
            return _build_cifar100_mapping_from_c2i(c2i)
        # Fallback if names unavailable
        return _build_cycle_mapping(num_classes)

    if name == "imagenet50":
        # Without a provided class name list for your ImageNet-50 subset,
        # default to a cycle; swap to a semantic mapping if you have wnids.
        return _build_cycle_mapping(num_classes)

    # Default fallback
    return _build_cycle_mapping(num_classes)

def _apply_asymmetric_exact(y: np.ndarray,
                            mapping: dict,
                            flip_indices: np.ndarray) -> np.ndarray:
    """
    Given ground-truth labels y and indices to flip, return noisy labels that
    map each selected label y[i] -> mapping[y[i]] (exactly |flip_indices| flips).
    """
    y_noisy = y.copy()
    # vectorized apply mapping
    y_noisy[flip_indices] = np.fromiter((mapping[int(c)] for c in y[flip_indices]),
                                        dtype=y.dtype, count=len(flip_indices))
    return y_noisy


# ------------------------------------------------------------
#  Results
# ------------------------------------------------------------

nu_hat_cifar100 = {
    100: {0.10: 0.57421875, 0.20: 0.39453125, 0.25: 0.24609375,
          0.33: 0.33203125, 0.50: 0.08984375},
    # 200: {0.10: 0.58984375, 0.20: 0.44140625, 0.25: 0.37890625,
    #       0.33: 0.36328125, 0.50: 0.26171875},
    300: {0.10: 0.53515625, 0.20: 0.41796875, 0.25: 0.41015625,
          0.33: 0.33984375, 0.50: 0.22265625},
    400: {0.10: 0.51171875, 0.20: 0.41796875, 0.25: 0.37109375,
          0.33: 0.37109375, 0.50: 0.27734375},
    1000: {0.1: -1, 0.20: 0.37109375, 0.25: 0.32421875,
           0.33: 0.26953125, 0.50: 0.22265625},
}

nu_hat_Imagenet50 = {
}

nu_hat = nu_hat_cifar100

# ------------------------------------------------------------
#  Active Learning sampling
# ------------------------------------------------------------

def get_train_idx(model: str, al_method: str, dataset: str, noise_type: str, budget: int, tau: float, annotator: str, train_idx: np.ndarray) -> np.ndarray:
    """
    return the lset that was sampled in the al experiment
    dataset in lowercase
    model in lowercase (resnet/linear)
    noise type (asym/sym)
    al_method (random/probcover or margin)

    """
    if al_method == 'random':
        return np.random.choice(train_idx, size=budget, replace=False)
    base_dir = "/cs/labs/daphna/tomer_yaacoby/pythonProjects/oracle-teacher-proj/experiments_data/"
    df = pd.read_parquet( base_dir + f"{dataset.upper()}/{dataset}_{model}_{al_method}_{noise_type}_all.parquet")
    alpha = 1 if annotator == 'oracle' else 0
    df_for_experiment = df[(df['budget'] == budget) & (df['oracle_teacher_ratio'] == tau) & (df['alpha'] == alpha)]
    # choose  train_idx randomly  from all the lsets
    run_idx = np.random.randint(0, len(df_for_experiment))
    lsets = df_for_experiment.iloc[run_idx]['lsets']
    train_idx = lsets[0]  # take the first lset
    return train_idx

# ------------------------------------------------------------
#  Core routine
# ------------------------------------------------------------
def estimate_nu(dataset: Dataset,
                tau: float,
                budget: int,
                num_classes: int,
                model_name: str = "resnet18",
                representation_model: str = 'simclr',
                epsilon: float = 1e-3,
                train_transform = None,
                test_transform = None,
                *,
                test_size: float = 0.1,
                batch_size: int = 32,
                epochs: int = 200,
                lr: float = 1e-3,
                seed: int = 42,
                cluster_labels=False,  # keep existing flag meaning: "use clusters"
                cluster_method: str = "kmeans",  # 'kmeans' | 'gmm' | 'agglomerative'
                runs: int = 1,
                noise_mode: str = "asymmetric",  # "random" or "asymmetric"
                al_method: str = "random",
                dataset_name: str = "",
                ) -> (int, float):
    """
    Estimate the noise fraction ν ∈ [0,1] by binary search.

    Parameters
    ----------
    dataset : torch.Tensor, shape (N, d)
        Feature matrix.
    tau : float
        Scaling factor used in the definition of |𝔻_noisy|.
    budget : int
        Oracle budget |𝔻_oracle|.
    num_classes : int
        Number of clusters / pseudo-labels.
    model_name : str, default "resnet18"
        Backbone to train ("resnet18" or "mlp").
    epsilon : float, default 1e-3
        Binary-search tolerance.
    Other kwargs control data-loader & optimiser behaviour.

    Returns
    -------
    nu_hat : float
        Estimated ν.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = dataset.features
    N, d = features.shape
    assert budget < N, "Budget must be smaller than dataset size N."
    if isinstance(features, torch.Tensor):
        x_np = features.cpu().numpy()
    else:
        x_np = features
    img_shape = dataset[0][0].shape

    # ------------------------------------------------------------------
    # 1) Unsupervised labelling via K-Means
    # ------------------------------------------------------------------
    if cluster_labels:
        labels = _cluster_labels(x_np, num_classes, method=cluster_method, seed=seed)
        if hasattr(dataset, 'noisy_labels'):
            dataset.noisy_labels = labels
        if hasattr(dataset, 'targets'):
            dataset.targets = labels.tolist()
    else:
        labels = np.array(dataset.targets)

    # labels: np.array of ground-truth for this run (original or KMeans)
    labels_are_kmeans = bool(cluster_labels)
    asym_mapping = _make_asym_mapping(
        cfg.DATASET.NAME if hasattr(cfg, "DATASET") else "",
        num_classes,
        dataset,
        labels_are_kmeans)

    # ------------------------------------------------------------------
    # 2) Train / test split + oracle subset
    # ------------------------------------------------------------------
    idxs = np.arange(N)
    train_idx, test_idx = train_test_split(idxs, test_size=test_size, random_state=seed, stratify=labels)
    print(f"Dataset size: {N}, Train size: {len(train_idx)}, Test size: {len(test_idx)}", flush=True)
    logger.info(f"Dataset size: {N}, Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Fixed test loader - run 5 times to get a stable accuracy
    test_copy = copy(dataset)
    if test_transform is not None:
        test_copy.transform = test_transform
    ds_test = Subset(test_copy, test_idx)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Oracle subset (size = budget, sampled without replacement)
    # repeat on training 5 times and take average accuracy as acc_oracle?
    train_copy = copy(dataset)
    if train_transform is not None:
        train_copy.transform = train_transform
    oracle_accs = []
    oracle_runs = 2
    for r in range(oracle_runs):
        # oracle_idx = np.random.choice(train_idx, size=budget, replace=False) # TODO change the al method of sampling
        oracle_idx = get_train_idx(model_name.lower(), al_method, dataset_name, 'sym', budget, tau, "oracle", train_idx)
        ds_oracle = Subset(dataset, oracle_idx)
        dl_oracle = DataLoader(ds_oracle, batch_size=batch_size, shuffle=True)

        m_oracle = _build_model(img_shape, num_classes, model_name, representation_model=representation_model)
        oracle_epochs = 100 if model_name == "linear" else epochs
        _train(m_oracle, dl_oracle, device, oracle_epochs, lr, cfg)
        oracle_accs.append(_accuracy(m_oracle, dl_test, device))

    acc_oracle = float(np.mean(oracle_accs))
    print(f"Oracle accuracy: {acc_oracle:.3f}", flush=True)
    logger.info(f"Oracle accuracy: {acc_oracle:.3f}")

    # Sample noisy annotator's indices
    noisy_size = int(budget / tau)
    noisy_idx = get_train_idx(model_name.lower(), al_method, dataset_name, 'sym', budget, tau, "noisy", train_idx)
    print(f"Noisy annotator size: {noisy_size}", flush=True)

    # ------------------------------------------------------------------
    # 3) Binary search over ν
    # ------------------------------------------------------------------

    nu_low, nu_high = 0.0, 1.0
    i = 0
    print(f"Oracle budget: {budget} noisy size: {noisy_size} ({budget}/{tau})", flush=True)
    logger.info(f"Oracle budget: {budget} noisy size: {noisy_size} ({budget}/{tau})")
    while (nu_high - nu_low) > epsilon:
        nu_mid = (nu_high + nu_low) / 2.0
        print(f"Binary search iteration {i+1}: nu_low={nu_low:.3f}, nu_mid={nu_mid:.3f}, nu_high={nu_high:.3f}", flush=True)
        logger.info(f"Binary search iteration {i+1}: nu_low={nu_low:.3f}, nu_mid={nu_mid:.3f}, nu_high={nu_high:.3f}")

        # sample noisy labels
        noisy_accs = []
        for r in range(runs):
            # --- sample noisy labels for this repetition -------------
            noisy_labels = labels.copy()
            noise_idx = np.random.choice(noisy_idx, size=int(nu_mid * noisy_size), replace=False)
            # all_labels = np.arange(num_classes)
            # valid = all_labels != noisy_labels[noise_idx, None]
            # noisy_labels[noise_idx] = np.apply_along_axis(
            #     lambda v: np.random.choice(all_labels[v]), 1, valid
            # ) # choose a random label that is not the correct one

            rng = np.random.default_rng(seed)  # or reuse your RNG
            all_labels = np.arange(num_classes)

            # choose behavior
            chooser = (
                (lambda y: rng.choice(all_labels[all_labels != y]))  # random wrong class
                if noise_mode == "random"
                else (lambda y: asym_mapping[int(y)])
            # asymmetric mapping: y -> mapping[y]
            )

            # apply to exactly the indices you picked to flip
            noisy_labels[noise_idx] = np.fromiter(
                (chooser(y) for y in noisy_labels[noise_idx]),
                dtype=noisy_labels.dtype,
                count=len(noise_idx)
            )

            if hasattr(dataset, "noisy_labels"):
                dataset.noisy_labels = noisy_labels
            if hasattr(dataset, "targets"):
                dataset.targets = noisy_labels.tolist()
            if hasattr(dataset, "labels"):
                dataset.labels = noisy_labels.tolist()



            ds_noisy = Subset(dataset, noisy_idx)
            dl_noisy = DataLoader(ds_noisy, batch_size=batch_size, shuffle=True)

            m_noisy = _build_model(img_shape, num_classes, model_name, representation_model=representation_model)
            _train(m_noisy, dl_noisy, device, epochs, lr, cfg)
            noisy_accs.append(_accuracy(m_noisy, dl_test, device))

        acc_noisy = float(np.mean(noisy_accs))

        # Binary-search update rule
        print(f"Oracle accuracy {acc_oracle}, Noisy accuracy: {acc_noisy:.3f}", end=" ", flush=True)
        logger.info(f"Oracle accuracy {acc_oracle}, Noisy accuracy: {acc_noisy:.3f}")
        if acc_oracle > acc_noisy:
            print("Oracle accuracy is higher than noisy accuracy, moving nu_high down.", flush=True)
            logger.info("Oracle accuracy is higher than noisy accuracy, moving nu_high down.")
            nu_high = nu_mid
        else:
            print("Noisy accuracy is higher than or equal to oracle accuracy, moving nu_low up.", flush=True)
            logger.info("Noisy accuracy is higher than or equal to oracle accuracy, moving nu_low up.")
            nu_low = nu_mid

        i += 1

    # Final estimate
    final_nu = (nu_high + nu_low) / 2.0
    return i, final_nu

def plot_results():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Patch

    # --- Data -------------------------------------------------------------------
    noise_levels = np.round(np.arange(0.1, 1.0, 0.1), 2)       # 0.1 … 0.9
    tau_values   = [0.10, 0.20, 0.25, 0.33, 0.50]              # rows (ascending)

    # nu_hat = {
    #     100: {0.10: 0.57421875, 0.20: 0.39453125, 0.25: 0.24609375,
    #           0.33: 0.33203125, 0.50: 0.08984375},
    #     # 200: {0.10: 0.58984375, 0.20: 0.44140625, 0.25: 0.37890625,
    #     #       0.33: 0.36328125, 0.50: 0.26171875},
    #     300: {0.10: 0.53515625, 0.20: 0.41796875, 0.25: 0.41015625,
    #           0.33: 0.33984375, 0.50: 0.22265625},
    #     400: {0.10: 0.51171875, 0.20: 0.41796875, 0.25: 0.37109375,
    #           0.33: 0.37109375, 0.50: 0.27734375},
    #     1000: {0.1: -1, 0.20: 0.37109375, 0.25: 0.32421875,
    #            0.33: 0.26953125, 0.50: 0.22265625},
    # }

    cmap = ListedColormap(["purple", "yellow"])

    # --- Plot --------------------------------------------------------------------
    for budget, tau_dict in nu_hat.items():
        # Build the indicator matrix (rows = tau, cols = noise)
        mat = np.zeros((len(tau_values), len(noise_levels)), dtype=int)
        for i, tau in enumerate(tau_values):
            nu = tau_dict[tau]
            for j, noise in enumerate(noise_levels):
                mat[i, j] = 1 if (noise - nu) > 0 else 0

        # Draw square, equally-sized cells
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
        im = ax.imshow(mat, cmap=cmap, origin="lower", aspect="equal")

        # Ticks in the middle of each cell
        ax.set_xticks(np.arange(len(noise_levels)))
        ax.set_xticklabels(noise_levels, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(tau_values)))
        ax.set_yticklabels(tau_values)

        # Thin grid lines to delineate cells
        ax.set_xticks(np.arange(-.5, len(noise_levels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(tau_values),    1), minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Annotate cell values in the centre
        for i in range(len(tau_values)):
            for j in range(len(noise_levels)):
                ax.text(j, i, str(np.round(tau_dict[tau_values[i]] - noise_levels[j], 2)),
                        ha="center", va="center", fontsize=7, color="black")

            # ============ ν̂ column on a NEW axis ============================
        divider = make_axes_locatable(ax)
        ax_nu = divider.append_axes("right", size="10%",pad=0.4)  # slim strip
        # nu_col = [tau_dict[t] for t in tau_values]  # → 1-D Python list
        # # make it a column vector for imshow
        # nu_img = np.array(nu_col)[:, None]  # shape (len(tau_values), 1)
        # ax_nu.imshow(nu_img, cmap="Greys", aspect="equal", vmin=0, vmax=1e-9)
        ax_nu.set_xticks([])
        ax_nu.set_yticks(np.arange(len(tau_values)))
        ax_nu.set_yticklabels([f"{nu_hat:.2f}" for nu_hat in tau_dict.values()], fontsize=9)
        ax_nu.tick_params(length=0, pad=2)
        ax_nu.set_ylabel(r"$\hat{\nu}$", rotation=270, labelpad=-20)
        for s in ax_nu.spines.values():
            s.set_visible(False)

        # ============ legend ============================================
        legend_elems = [
            Patch(facecolor='purple', label='noisy better'),
            Patch(facecolor='yellow', label='oracle better')
        ]
        leg = ax.legend(handles=legend_elems,
                        loc='upper right', fontsize=8, frameon=True)
        leg.get_frame().set_facecolor('#f2f2f2')
        leg.get_frame().set_edgecolor('none')

        ax.set_xlabel("noise level")
        ax.set_ylabel("cost ratios (τ)")
        ax.set_title(f"budget = {budget}")
        plt.tight_layout()
        plt.show()


def plot_results_tau():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Patch

    # ── ν̂  (K-Means) numbers you provided ────────────────────────────────
    # nu_hat = {
    #     100: {0.10: 0.57421875, 0.20: 0.39453125, 0.25: 0.24609375,
    #           0.33: 0.33203125, 0.50: 0.08984375},
    #     # 200: {0.10: 0.58984375, 0.20: 0.44140625, 0.25: 0.37890625,
    #     #       0.33: 0.36328125, 0.50: 0.26171875},
    #     300: {0.10: 0.53515625, 0.20: 0.41796875, 0.25: 0.41015625,
    #           0.33: 0.33984375, 0.50: 0.22265625},
    #     400: {0.10: 0.51171875, 0.20: 0.41796875, 0.25: 0.37109375,
    #           0.33: 0.37109375, 0.50: 0.27734375},
    #     1000: {0.1: -1, 0.20: 0.37109375, 0.25: 0.32421875,
    #             0.33: 0.26953125, 0.50: 0.22265625},
    # }

    tau_vals = [0.10, 0.20, 0.25, 0.33, 0.50]  # rows (ascending)
    budgets = nu_hat.keys()  # budgets = [100, 200, 300, 400, 1000]
    noise_levels = np.round(np.arange(0.1, 1.0, 0.1), 2)  # 0.1 … 0.9
    cmap = ListedColormap(["purple", "yellow"])

    # ── Build & plot one heat-table for each τ ────────────────────────────
    for tau in tau_vals:
        # Indicator matrix (rows = budget, cols = noise)
        mat = np.zeros((len(budgets), len(noise_levels)), dtype=int)
        for i, budget in enumerate(budgets):
            nu = nu_hat[budget][tau]
            for j, noise in enumerate(noise_levels):
                mat[i, j] = 1 if (nu-noise) < 0 else 0

        # X-tick labels are τ − (1 − noise)
        x_labels = np.round(tau - (1 - noise_levels), 2)

        # ── Figure ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
        im = ax.imshow(mat, cmap=cmap, origin="lower", aspect="equal")

        # ticks centred on each cell
        ax.set_xticks(np.arange(len(noise_levels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(budgets)))
        ax.set_yticklabels(budgets)

        # put a red vertical line at p = 0
        p_vals = [tau - (1 - noise) for noise in noise_levels]
        if 0 in p_vals:
            ax.axvline(p_vals.index(0), color='red', linewidth=2)
        else:
            for k in range(1, len(p_vals)):
                if (p_vals[k - 1] < 0 < p_vals[k]) or (
                        p_vals[k - 1] > 0 > p_vals[k]):
                    ax.axvline(k - 0.5, color='red', linewidth=2)
                    break

        # thin grid around every cell
        ax.set_xticks(np.arange(-.5, len(noise_levels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(budgets), 1), minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # centre the “0” / “1” annotations
        for i in range(len(budgets)):
            for j in range(len(noise_levels)):
                ax.text(j, i, str(np.round(nu_hat[list(budgets)[i]][tau] - noise_levels[j], 2)),
                        ha="center", va="center", fontsize=7, color="black")

        # ---------------------------------------------------------------
        # 1️⃣  RIGHT-HAND  ν̂ COLUMN  (closer, narrow strip)
        # ---------------------------------------------------------------
        divider = make_axes_locatable(ax)
        ax_nu = divider.append_axes("right", size="10%",
                                    pad=0.4)  # ← pad small
        ax_nu.set_ylim(ax.get_ylim())
        ax_nu.set_xticks([])  # no x-ticks
        ax_nu.set_yticks(np.arange(len(budgets)))
        ax_nu.set_yticklabels([f"{nu_hat[B][tau]:.2f}" for B in budgets],
                              fontsize=8)
        ax_nu.tick_params(
            axis='y',
            which='both',
            length=0,  # ← no tick mark
            pad=2  # ← small gap between text and heat-map
        )

        ax_nu.set_ylabel(r"$\hat{\nu}$", fontsize=10, rotation=270,
                         labelpad=-20)
        for spine in ax_nu.spines.values():  # no rectangle
            spine.set_visible(False)

        # ---------------------------------------------------------------
        # 2️⃣  SMALL  LEGEND  WITH  GREY  BACKGROUND
        # ---------------------------------------------------------------
        legend_elems = [
            Patch(facecolor='purple', label='noisy better'),
            Patch(facecolor='yellow', label='oracle better')
        ]
        leg = ax.legend(handles=legend_elems,
                        loc='upper right',
                        fontsize=8,
                        frameon=True)
        leg.get_frame().set_facecolor('#f2f2f2')  # light-grey
        leg.get_frame().set_edgecolor('none')  # no border

        # remove outer spines around heat-map
        # for spine in ax.spines.values():
        #     spine.set_visible(False)

        ax.set_xlabel(r"$\tau - (1-\mathrm{noise})$")
        ax.set_ylabel("budget")
        ax.set_title(fr"Indicator map  (fixed $\tau={tau}$)")
        plt.tight_layout()

        # Save the figure
        results_dir = "results/tau_plots"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"indicator_map_tau_{tau}.png"))

        plt.show()


def run_estimated_nu():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["cifar10", "cifar100", "imagenet50"],
                        default="imagenet50")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", choices=["resnet18", "vit", "linear", "resnet50"],
                        default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--al_method", choices=["random", "margin", "probcover"],
                        default="probcover")
    parser.add_argument("--noise_mode", choices=["random", "asymmetric"],
                        default="random")
    parser.add_argument("--cluster_method", choices=["kmeans", "gmm", "agglomerative", "agglomerative_fast", "birch", "dbscan"],
                        default="kmeans")
    parser.add_argument("--dbscan_eps", type=float, default=None, help="DBSCAN eps; if omitted we'll auto-tune to target num_classes")
    parser.add_argument("--dbscan_min_samples", type=int, default=10)
    parser.add_argument("--dbscan_metric", type=str, default="euclidean", choices=["euclidean", "manhattan", "cosine"])
    parser.add_argument("--representation_model", type=str, default="simclr", choices=["simclr", "vit", "resnet50"])
    args = parser.parse_args()

    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    add_path(os.path.abspath(
        '/cs/labs/daphna/tomer_yaacoby/pythonProjects/oracle-teacher-proj/TypiClust/deep-al'))  # TODO change properly

    def get_dataset(dataset_name, representation_model, train=True,
                    use_linear_features=False, project_features=False):
        cfg_file = f"/cs/labs/daphna/tomer_yaacoby/pythonProjects/oracle-teacher-proj/TypiClust/deep-al/configs/{dataset_name}/al/RESNET18.yaml"
        cfg.merge_from_file(cfg_file)
        # cfg.NOISE.NOISE_RATE = 0
        # cfg.DATASET.REPRESENTATION_MODEL = representation_model
        cfg.MODEL.LINEAR_FROM_FEATURES = use_linear_features
        data_obj = Data(cfg)
        data, data_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR,
                                              representation_model=representation_model,
                                              isTrain=train, isDownload=True,
                                              project_features=project_features,
                                              )
        return data, data_size

    # todo add parse arguments for dataset, budget, epochs
    seed = args.seed
    dataset_name = args.dataset
    rep = args.representation_model
    project_features = False
    model_name = args.model
    # tau = 0.25
    budget = args.budget
    test_ratio = 0.2
    epsilon = 0.01

    _set_seed(seed)
    print(f"Setting seed to {seed}", flush=True)
    # epochs = 500 if model_name == 'linear' else args.epochs # 200 for resnet (09/13/2025)
    epochs = args.epochs
    use_linear_features = model_name == 'linear'
    num_classes_dict = {"cifar10": 10, "cifar100": 100, "imagenet50": 50}
    num_classes = num_classes_dict[dataset_name]

    # Load dataset
    print(f"Loading dataset {dataset_name}", flush=True)
    logger.info(f"Loading dataset {dataset_name}")
    train_data, train_size = get_dataset(dataset_name, rep,
                                         train=True,
                                         use_linear_features=use_linear_features,
                                         project_features=project_features)

    tau_vals = [args.tau]  # [0.1, 0.2, 0.25, 0.33, 0.5]
    nu_hats_original_labels = []
    num_tries_original_labels = []
    nu_hats_kmeans_labels = []
    num_tries_kmeans_labels = []
    batch_size = 256 if dataset_name != 'imagenet50' else 16
    for tau in tau_vals:
        print(
            f"Estimating nu for tau={tau} with budget={budget} and model={model_name}",
            flush=True)
        logger.info(
            f"Estimating nu for tau={tau} with budget={budget} and model={model_name}")
        print(f"Using noise mode: {args.noise_mode}", flush=True)
        logger.info(f"Using noise mode: {args.noise_mode}")
        print(f"Using clustering method: {args.cluster_method}", flush=True)
        logger.info(f"Using clustering method: {args.cluster_method}")
        print(f"Using representation model: {rep}", flush=True)
        logger.info(f"Using representation model: {rep}")
        print(f"Using AL method: {args.al_method}", flush=True)
        logger.info(f"Using AL method: {args.al_method}")
        # i, nu_hat_original_labels = estimate_nu(train_data,
        #                                         tau=tau,
        #                                         budget=budget,
        #                                         num_classes=num_classes,
        #                                         model_name=model_name,
        #                                         epsilon=epsilon,
        #                                         epochs=epochs,
        #                                         test_size=test_ratio,
        #                                         batch_size=batch_size,
        #                                         seed=seed,
        #                                         kmeans_labels=False)
        # print(
        #     f"Estimated nu for tau={tau}: {nu_hat_original_labels:.4f}, tries: {i}",
        #     flush=True)
        # logger.info(
        #     f"Estimated nu for tau={tau}: {nu_hat_original_labels:.4f}, num tries: {i}")
        # nu_hats_original_labels.append(nu_hat_original_labels)
        # num_tries_original_labels.append(i)

        i, nu_hat_cluster_labels = estimate_nu(train_data,
                                              tau=tau,
                                              budget=budget,
                                              num_classes=num_classes,
                                              model_name=model_name,
                                              representation_model=rep,
                                              epsilon=epsilon,
                                              epochs=epochs,
                                              test_size=test_ratio,
                                              seed=seed,
                                              cluster_labels=True,
                                              cluster_method=args.cluster_method,
                                              noise_mode=args.noise_mode,
                                              al_method = args.al_method,
                                              dataset_name=dataset_name
                                              )
        print(
            f"Estimated nu for tau={tau} with Clustered labels: {nu_hat_cluster_labels:.4f}, tries: {i}",
            flush=True)
        logger.info(
            f"Estimated nu for tau={tau} with Clustered labels: {nu_hat_cluster_labels:.4f}, num tries: {i}")
        nu_hats_kmeans_labels.append(nu_hat_cluster_labels)
        num_tries_kmeans_labels.append(i)

    print("Tau values:", tau_vals, flush=True)
    logger.info(f"Tau values: {tau_vals}")
    # print("Estimated nu for original labels:", nu_hats_original_labels,
    #       flush=True)
    # logger.info(f"Estimated nu for original labels: {nu_hats_original_labels}")
    # print("Number of tries for original labels:", num_tries_original_labels,
    #       flush=True)
    # logger.info(
    #     f"Number of tries for original labels: {num_tries_original_labels}")
    print(f"Estimated nu for Clustered labels:", nu_hats_kmeans_labels, flush=True)
    logger.info(f"Estimated nu for Clustered labels: {nu_hats_kmeans_labels}")
    print(f"Number of tries for Clustered labels:", num_tries_kmeans_labels,
          flush=True)
    logger.info(
        f"Number of tries for Clustered labels: {num_tries_kmeans_labels}")

    # Save results
    results_dir = f"results/{dataset_name}/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"nu_estimates_{rep}_1.txt")
    with open(results_file, 'w') as f:
        f.write("Tau values: " + str(tau_vals) + "\n")
        # f.write("Estimated nu for original labels: " + str(
        #     nu_hats_original_labels) + "\n")
        # f.write("Number of tries for original labels: " + str(
        #     num_tries_original_labels) + "\n")
        f.write(f"Estimated nu for {args.cluster_method} labels: " + str(
            nu_hats_kmeans_labels) + "\n")
        f.write("Number of tries for KMeans labels: " + str(
            num_tries_kmeans_labels) + "\n")


# ------- Main - only a test script to make sure the function works -------

if __name__ == "__main__":
    from pycls.core.config import cfg
    from pycls.datasets.data import Data
    from pycls.core.builders import FeaturesNet

    run_estimated_nu()
    # plot_results()
    # plot_results_tau()
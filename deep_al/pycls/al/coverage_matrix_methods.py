import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.al.kernel_utils import build_K_general_matrix, RBFKernel, TopHatKernel

torch.cuda.empty_cache()


class CoverageMatrixMethod:
    def __init__(self, cfg, train_labels):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        self.all_features = ds_utils.load_features(self.ds_name, train=True, features_tag=features_tag)
        self.debug = self.cfg.DEBUG
        self.use_sparse = getattr(self.cfg, 'SPARSE_K', False)
        self.device = 'cuda' if (self.use_sparse or self.all_features.shape[0] <= 100000) else 'cpu'

        mat_type = getattr(self.cfg, 'MATRIX_DTYPE', 'float32')
        self.matrices_type = getattr(torch, mat_type)

        self.features_gpu = torch.from_numpy(self.all_features).to(self.device, dtype=self.matrices_type)
        self.budgetSize = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0
        self.sampling_fn = str(cfg.ACTIVE_LEARNING.SAMPLING_FN).lower().replace('-', '_')
        self.is_npc = (
            self.sampling_fn.startswith('npc_probcover')
            or self.sampling_fn.startswith('npc_maxherding')
        )
        self.nas_filter_method = str(getattr(cfg.ACTIVE_LEARNING, 'NAS_FILTER_METHOD', 'none')).lower()
        self.last_npc_filter_stats = None
        self.npc_filter_history = []

        self.alpha = 0
        if cfg.ACTIVE_LEARNING.INITIAL_DELTA < 0:
            print("Delta not specified. Starting automated selection process...")

            # Track time for performance monitoring
            start_time = time.time()

            print("Running Delta selection algorithm...")

            cfg.ACTIVE_LEARNING.INITIAL_DELTA = ds_utils.select_delta(
                features=self.all_features,
                num_classes=self.cfg.MODEL.NUM_CLASSES,
                seed=self.seed
            )

            elapsed = time.time() - start_time
            print(f"Delta selection complete | Value: {cfg.ACTIVE_LEARNING.INITIAL_DELTA:.4f} | Time: {elapsed:.2f}s")

        self.delta = cfg.ACTIVE_LEARNING.INITIAL_DELTA

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.unique_labels = unique_labels
        self.num_of_classes = unique_labels.size

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
        else:
            self.kernel_fn = RBFKernel('cuda')
            initial_threshold = np.exp(-1.0 * (self.delta / self.sigma) ** 2) if self.use_sparse else 0.0

        self.K_general = build_K_general_matrix(
            self.all_features,
            threshold=initial_threshold,
            use_sparse=self.use_sparse,
            kernel_type=self.kernel_type,
            delta=self.delta,
            sigma=self.sigma,
            kernel_build_batch_size=self.kernel_build_batch_size,
            matrices_type=self.matrices_type,
            kernel_fn=self.kernel_fn,
            zero_indices=None,
        )

        # Coverage is rebuilt from the current labeled set at every AL episode.
        # Keeping a persistent coverage matrix made standard MaxHerding/ProbCover
        # behave differently from their NPC variants even when NPC filtered nothing.

    def _get_dense_kernel_row(self, position):
        row = self.K[int(position)]
        if self.use_sparse:
            row = row.to_dense()
        return row.squeeze().to(device=self.device, dtype=self.matrices_type)

    def _update_class_agnostic_coverage(self, kernel_row):
        """Update coverage without using the unknown label of an unlabeled point.

        The selection objective depends only on the maximum coverage over classes.
        We therefore store all class-agnostic coverage in column 0. This preserves
        the previous score computation while removing label leakage.
        """
        self.C[:, 0] = torch.maximum(self.C[:, 0], kernel_row)

    def init_sampling_loop(self, lset, uset, clean_lset_pos=None):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []

        if self.use_sparse:
            K_csr_shuffled = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
            crow_indices = torch.from_numpy(K_csr_shuffled.indptr).to(torch.int64)
            col_indices = torch.from_numpy(K_csr_shuffled.indices).to(torch.int64)
            values = torch.from_numpy(K_csr_shuffled.data).to(torch.float32)

            self.K = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=values,
                size=K_csr_shuffled.shape,
                dtype=values.dtype
            )
            del K_csr_shuffled, values, col_indices, crow_indices
        else:
            self.K = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
        self.K = self.K.to(self.device) if self.K is not None else None
        self.train_labels = self.train_labels_general[self.relevant_indices]

        # Rebuild coverage from scratch for both the baseline and NPC paths.
        # relevant_indices starts with lSet, so labeled positions are 0..len(lSet)-1.
        self.C = torch.full(
            (len(self.relevant_indices), self.num_of_classes),
            self.alpha,
            device=self.device,
            dtype=self.matrices_type,
        )

        if clean_lset_pos is None:
            # Standard ProbCover/MaxHerding: every labeled sample contributes.
            covering_lset_pos = np.arange(len(self.lSet), dtype=int)
        else:
            # NPC: only samples predicted as clean contribute.
            covering_lset_pos = np.asarray(clean_lset_pos, dtype=int)

        if covering_lset_pos.size:
            if np.any(covering_lset_pos < 0) or np.any(covering_lset_pos >= len(self.lSet)):
                raise ValueError(
                    "clean_lset_pos must contain positions inside the current labeled set"
                )
            for pos in covering_lset_pos:
                kernel_row = self._get_dense_kernel_row(int(pos))
                self._update_class_agnostic_coverage(kernel_row)
                del kernel_row

    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        # print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])

    @staticmethod
    def _l2_normalize_np(x, eps=1e-12):
        x = np.asarray(x, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

    @staticmethod
    def _get_current_labels_from_dataset(trainDataset):
        if hasattr(trainDataset, "noisy_labels"):
            return np.asarray(trainDataset.noisy_labels, dtype=np.int64)
        if hasattr(trainDataset, "targets"):
            return np.asarray(trainDataset.targets, dtype=np.int64)
        if hasattr(trainDataset, "dataset") and hasattr(trainDataset.dataset, "targets"):
            return np.asarray(trainDataset.dataset.targets, dtype=np.int64)
        raise RuntimeError("Could not read current labels from trainDataset")

    @staticmethod
    def _get_clean_labels_from_dataset(trainDataset):
        if hasattr(trainDataset, "clean_labels"):
            return np.asarray(trainDataset.clean_labels, dtype=np.int64)
        return None

    def _npc_stats(self, lset, noisy_mask, method):
        lset = np.asarray(lset, dtype=int)
        noisy_mask = np.asarray(noisy_mask, dtype=bool)
        stats = {
            "method": method,
            "labeled_size": int(len(lset)),
            "pred_noisy": int(noisy_mask.sum()),
            "pred_clean": int(len(lset) - noisy_mask.sum()),
            "predicted_noise_rate": float(noisy_mask.mean()) if len(noisy_mask) else 0.0,
        }
        return stats

    def _apply_noise_dropout(self, noisy_mask, lset):
        """Apply the NAS noise-dropout rule from Shafir et al.

        Let q_hat be the predicted noise rate and
            eta = max(min(q_hat, 1 - q_hat), min_fraction).
        Randomly restore eta of the samples predicted as noisy to the clean set
        for query selection only. Labels and the training set are not changed.
        """
        noisy_mask = np.asarray(noisy_mask, dtype=bool).copy()
        enabled = bool(getattr(self.cfg.ACTIVE_LEARNING, "NAS_NOISE_DROPOUT", True))
        min_fraction = float(
            getattr(self.cfg.ACTIVE_LEARNING, "NAS_NOISE_DROPOUT_MIN_FRACTION", 0.1)
        )
        min_fraction = float(np.clip(min_fraction, 0.0, 1.0))

        before = int(noisy_mask.sum())
        q_hat = float(noisy_mask.mean()) if len(noisy_mask) else 0.0
        info = {
            "noise_dropout_enabled": enabled,
            "pred_noisy_before_dropout": before,
            "predicted_noise_rate_before_dropout": q_hat,
            "noise_dropout_eta": 0.0,
            "noise_dropout_restored": 0,
        }

        if (not enabled) or before == 0:
            info["pred_noisy_after_dropout"] = before
            info["predicted_noise_rate_after_dropout"] = q_hat
            return noisy_mask, info

        eta = float(max(min(q_hat, 1.0 - q_hat), min_fraction))
        eta = float(np.clip(eta, 0.0, 1.0))
        n_restore = int(round(eta * before))
        n_restore = min(max(n_restore, 0), before)

        if n_restore > 0:
            noisy_positions = np.flatnonzero(noisy_mask)
            # Deterministic per seed and AL query while still varying by episode.
            query_index = len(self.npc_filter_history)
            rng_seed = int(self.seed or 0) + 1009 * int(len(lset)) + 7919 * query_index
            rng = np.random.default_rng(rng_seed)
            restored = rng.choice(noisy_positions, size=n_restore, replace=False)
            noisy_mask[restored] = False

        after = int(noisy_mask.sum())
        info.update({
            "noise_dropout_eta": eta,
            "noise_dropout_restored": int(before - after),
            "pred_noisy_after_dropout": after,
            "predicted_noise_rate_after_dropout": (
                float(noisy_mask.mean()) if len(noisy_mask) else 0.0
            ),
        })
        return noisy_mask, info

    def _crossval_noisy_mask(self, trainDataset, lset):
        """Netta-style CrossValidation filter.

        Train k linear classifiers on SSL features. A sample is suspected noisy if fewer
        than half of the models predict its current label.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold

        lset = np.asarray(lset, dtype=int)
        y_all = self._get_current_labels_from_dataset(trainDataset)
        y = y_all[lset]
        x = self._l2_normalize_np(self.all_features[lset])

        n = len(lset)
        folds = int(max(2, min(getattr(self.cfg.ACTIVE_LEARNING, "NAS_CV_FOLDS", 3), n)))
        max_iter = int(getattr(self.cfg.ACTIVE_LEARNING, "NAS_CV_MAX_ITER", 300))

        if n < 2 or len(np.unique(y)) < 2:
            return np.zeros(n, dtype=bool), {"reason": "too_few_samples_or_classes"}

        agree = np.zeros(n, dtype=np.int64)
        valid_models = 0
        splitter = KFold(n_splits=folds, shuffle=True, random_state=int(self.seed or 0))
        for train_pos, _ in splitter.split(x):
            y_train = y[train_pos]
            if len(np.unique(y_train)) < 2:
                continue
            clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", n_jobs=-1)
            clf.fit(x[train_pos], y_train)
            pred = clf.predict(x)
            agree += (pred.astype(np.int64) == y)
            valid_models += 1

        if valid_models == 0:
            return np.zeros(n, dtype=bool), {"reason": "no_valid_cv_models"}

        noisy = agree < (valid_models / 2.0)
        return noisy.astype(bool), {
            "valid_models": int(valid_models),
            "agree_min": int(agree.min()),
            "agree_max": int(agree.max()),
            "agree_mean": float(agree.mean()),
        }

    def _knn_noisy_mask(self, trainDataset, lset):
        """Appendix-style kNN filter over labeled SSL features."""
        from sklearn.neighbors import NearestNeighbors

        lset = np.asarray(lset, dtype=int)
        y_all = self._get_current_labels_from_dataset(trainDataset)
        y = y_all[lset]
        x = self._l2_normalize_np(self.all_features[lset])

        n = len(lset)
        if n <= 1:
            return np.zeros(n, dtype=bool), {"reason": "too_few_samples"}

        k_cfg = int(getattr(self.cfg.ACTIVE_LEARNING, "NAS_KNN_K", -1))
        if k_cfg <= 0:
            k = max(1, int(n // max(1, int(self.cfg.MODEL.NUM_CLASSES))))
        else:
            k = k_cfg
        k = int(min(k, n - 1))

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn.fit(x)
        _, neigh = nn.kneighbors(x, return_distance=True)

        noisy = np.zeros(n, dtype=bool)
        for i in range(n):
            neigh_i = neigh[i, 1:]  # remove self
            votes = y[neigh_i]
            if len(votes) == 0:
                continue
            majority = Counter(votes.tolist()).most_common(1)[0][0]
            noisy[i] = int(majority) != int(y[i])

        return noisy, {"k": int(k)}

    def _oracle_noisy_mask(self, trainDataset, lset):
        """Diagnostic upper-bound filter: uses the true clean labels.

        This is not a valid AL method. It is only an ablation to verify whether
        the NPC/ProbCover logic would help if the noise detector were perfect.
        """
        lset = np.asarray(lset, dtype=int)
        clean_labels = self._get_clean_labels_from_dataset(trainDataset)
        if clean_labels is None:
            return np.zeros(len(lset), dtype=bool), {"reason": "oracle_requested_but_clean_labels_missing"}

        current = self._get_current_labels_from_dataset(trainDataset)[lset]
        noisy = current != clean_labels[lset]
        return noisy.astype(bool), {
            "reason": "oracle_true_noisy_mask_for_diagnostics_only",
            "oracle_detector": True,
        }

    def _lowbudget_aum_noisy_mask(self, trainDataset, lset, uset):
        """Low-budget AUM-style filter on frozen SSL features with a fake class.

        The raw fake-class threshold can be extremely aggressive when there are
        about one labeled example per class. To avoid collapsing NPC coverage,
        NAS_AUM_MAX_PRED_NOISY_RATE caps the fraction of labeled samples that
        AUM may mark as noisy before noise dropout. Set the cap to 1.0 to recover
        the uncapped fake-percentile rule.
        """
        lset = np.asarray(lset, dtype=int)
        uset = np.asarray(uset, dtype=int)
        y_all = self._get_current_labels_from_dataset(trainDataset)
        y_l = y_all[lset]
        x_l = self._l2_normalize_np(self.all_features[lset])

        n = len(lset)
        if n < 2 or len(np.unique(y_l)) < 2 or len(uset) == 0:
            return np.zeros(n, dtype=bool), {"reason": "too_few_samples_classes_or_unlabeled"}

        fake_ratio = float(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_FAKE_RATIO", 1.0))
        fake_ratio = max(0.0, fake_ratio)
        n_fake = int(max(1, min(len(uset), round(fake_ratio * n))))
        rng = np.random.default_rng(int(self.seed or 0) + 1337 + n)
        fake_idx = rng.choice(uset, size=n_fake, replace=False)
        x_fake = self._l2_normalize_np(self.all_features[fake_idx])

        num_classes = int(self.cfg.MODEL.NUM_CLASSES)
        fake_class = num_classes

        x = np.concatenate([x_l, x_fake], axis=0).astype(np.float32)
        y = np.concatenate([y_l, np.full(n_fake, fake_class, dtype=np.int64)], axis=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = int(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_EPOCHS", 40))
        aum_seed = int(self.seed or 0) + 4242 + n

        # select_samples() intentionally runs under torch.inference_mode().
        # LowBudgetAUM is the one part that must train a linear classifier, so
        # gradients are locally re-enabled without changing the rest of sampling.
        fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(aum_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(aum_seed)

            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)

            model = nn.Linear(xt.shape[1], num_classes + 1).to(device)
            opt = torch.optim.SGD(
                model.parameters(),
                lr=float(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_LR", 0.1)),
                weight_decay=float(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_WEIGHT_DECAY", 0.0)),
            )
            margin_sum = torch.zeros(xt.shape[0], device=device)

            for _ in range(max(1, epochs)):
                logits = model(xt)
                with torch.no_grad():
                    true_logits = logits.gather(1, yt.view(-1, 1)).squeeze(1)
                    other_logits = logits.detach().clone()
                    other_logits[torch.arange(yt.numel(), device=device), yt] = -float("inf")
                    max_other = other_logits.max(dim=1).values
                    margin_sum += (true_logits.detach() - max_other)

                loss = F.cross_entropy(logits, yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            aum = (margin_sum / float(max(1, epochs))).detach().cpu().numpy()

        labeled_aum = aum[:n]
        fake_aum = aum[n:]
        percentile = float(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_FAKE_PERCENTILE", 80.0))
        raw_threshold = float(np.percentile(fake_aum, percentile))

        max_rate = float(getattr(self.cfg.ACTIVE_LEARNING, "NAS_AUM_MAX_PRED_NOISY_RATE", 1.0))
        max_rate = float(np.clip(max_rate, 0.0, 1.0))
        if max_rate < 1.0:
            cap_threshold = float(np.percentile(labeled_aum, 100.0 * max_rate))
            threshold = float(min(raw_threshold, cap_threshold))
            threshold_was_capped = threshold < raw_threshold
        else:
            cap_threshold = None
            threshold = raw_threshold
            threshold_was_capped = False

        noisy_uncapped = labeled_aum <= raw_threshold
        noisy = labeled_aum <= threshold

        return noisy.astype(bool), {
            "epochs": int(epochs),
            "n_fake": int(n_fake),
            "fake_ratio": float(fake_ratio),
            "fake_percentile": float(percentile),
            "threshold_raw_fake": raw_threshold,
            "threshold_cap_labeled": None if cap_threshold is None else float(cap_threshold),
            "threshold": float(threshold),
            "threshold_was_capped": bool(threshold_was_capped),
            "aum_max_pred_noisy_rate": float(max_rate),
            "aum_uncapped_pred_noisy": int(noisy_uncapped.sum()),
            "aum_uncapped_predicted_noise_rate": float(noisy_uncapped.mean()),
            "labeled_aum_mean": float(labeled_aum.mean()),
            "labeled_aum_std": float(labeled_aum.std()),
            "fake_aum_mean": float(fake_aum.mean()),
            "fake_aum_std": float(fake_aum.std()),
        }

    def _add_oracle_quality_stats(self, stats, raw_noisy_mask, effective_noisy_mask, true_noisy):
        """Add precision/recall/F1 diagnostics for detector masks."""
        true_noisy = np.asarray(true_noisy, dtype=bool)
        masks = {
            "before_dropout": np.asarray(raw_noisy_mask, dtype=bool),
            "after_dropout": np.asarray(effective_noisy_mask, dtype=bool),
        }
        stats["oracle_true_noisy_rate"] = float(true_noisy.mean()) if len(true_noisy) else 0.0
        stats["oracle_true_clean"] = int((~true_noisy).sum())

        for suffix, mask in masks.items():
            tp = int((mask & true_noisy).sum())
            fp = int((mask & (~true_noisy)).sum())
            fn = int(((~mask) & true_noisy).sum())
            tn = int(((~mask) & (~true_noisy)).sum())
            precision = float(tp / max(1, tp + fp))
            recall = float(tp / max(1, tp + fn))
            f1 = float((2.0 * precision * recall) / max(1e-12, precision + recall))
            false_positive_rate = float(fp / max(1, fp + tn))
            stats[f"oracle_precision_{suffix}"] = precision
            stats[f"oracle_recall_{suffix}"] = recall
            stats[f"oracle_f1_{suffix}"] = f1
            stats[f"oracle_false_positive_rate_{suffix}"] = false_positive_rate

    def _npc_clean_lset_positions(self, trainDataset, lset, uset):
        method = str(self.nas_filter_method or "none").lower()
        lset = np.asarray(lset, dtype=int)
        min_labeled = int(getattr(self.cfg.ACTIVE_LEARNING, "NAS_MIN_LABELED_SIZE", 0))

        if len(lset) == 0 or method in ["none", ""]:
            noisy_mask = np.zeros(len(lset), dtype=bool)
            extra = {"reason": "empty_or_no_filter"}
        elif method != "oracle" and min_labeled > 0 and len(lset) < min_labeled:
            noisy_mask = np.zeros(len(lset), dtype=bool)
            extra = {
                "reason": "nas_delayed_until_min_labeled_size",
                "nas_min_labeled_size": int(min_labeled),
            }
        elif method == "crossval":
            noisy_mask, extra = self._crossval_noisy_mask(trainDataset, lset)
        elif method == "knn":
            noisy_mask, extra = self._knn_noisy_mask(trainDataset, lset)
        elif method == "aum":
            noisy_mask, extra = self._lowbudget_aum_noisy_mask(trainDataset, lset, uset)
        elif method == "oracle":
            noisy_mask, extra = self._oracle_noisy_mask(trainDataset, lset)
        else:
            raise ValueError(f"Unknown NAS_FILTER_METHOD={method}")

        raw_noisy_mask = np.asarray(noisy_mask, dtype=bool).copy()
        if method == "oracle":
            # Oracle-NPC is an upper-bound diagnostic. Do not weaken the perfect
            # detector by applying random noise dropout.
            noisy_mask = raw_noisy_mask.copy()
            q_hat = float(noisy_mask.mean()) if len(noisy_mask) else 0.0
            dropout_info = {
                "noise_dropout_enabled": False,
                "noise_dropout_reason": "disabled_for_oracle_upper_bound",
                "pred_noisy_before_dropout": int(raw_noisy_mask.sum()),
                "predicted_noise_rate_before_dropout": q_hat,
                "noise_dropout_eta": 0.0,
                "noise_dropout_restored": 0,
                "pred_noisy_after_dropout": int(noisy_mask.sum()),
                "predicted_noise_rate_after_dropout": q_hat,
            }
        else:
            noisy_mask, dropout_info = self._apply_noise_dropout(raw_noisy_mask, lset)

        clean_pos = np.where(~noisy_mask)[0].astype(int)
        stats = self._npc_stats(lset, noisy_mask, method)
        stats.update(extra or {})
        stats.update(dropout_info)
        stats["query_index"] = int(len(self.npc_filter_history))
        stats["unlabeled_size"] = int(len(uset))
        stats["nas_min_labeled_size"] = int(min_labeled)

        clean_labels = self._get_clean_labels_from_dataset(trainDataset)
        if clean_labels is not None and len(lset) > 0:
            current = self._get_current_labels_from_dataset(trainDataset)[lset]
            true_noisy = current != clean_labels[lset]
            stats.update({
                # Detector quality before noise dropout.
                "oracle_true_noisy": int(true_noisy.sum()),
                "oracle_correctly_flagged_before_dropout": int(
                    (raw_noisy_mask & true_noisy).sum()
                ),
                "oracle_clean_flagged_before_dropout": int(
                    (raw_noisy_mask & (~true_noisy)).sum()
                ),
                "oracle_missed_noisy_before_dropout": int(
                    ((~raw_noisy_mask) & true_noisy).sum()
                ),
                # Effective NPC mask after noise dropout.
                "oracle_correctly_flagged": int((noisy_mask & true_noisy).sum()),
                "oracle_clean_flagged": int((noisy_mask & (~true_noisy)).sum()),
                "oracle_missed_noisy": int(((~noisy_mask) & true_noisy).sum()),
            })
            self._add_oracle_quality_stats(stats, raw_noisy_mask, noisy_mask, true_noisy)

        self.last_npc_filter_stats = dict(stats)
        self.npc_filter_history.append(dict(stats))
        print(f"NPC filter stats: {stats}")
        return clean_pos

    @torch.inference_mode()
    def select_samples(self, lset, uset, trainDataset=None, checkpoint_file=None):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """

        clean_lset_pos = None
        if self.is_npc:
            if trainDataset is None:
                raise RuntimeError('NPC selection requires trainDataset for noise filtering')
            clean_lset_pos = self._npc_clean_lset_positions(trainDataset, lset, uset)
        self.init_sampling_loop(lset, uset, clean_lset_pos=clean_lset_pos)

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        for i in tqdm(range(self.budgetSize), desc="Selecting Samp"):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            # Coverage is class-agnostic and stored in column 0.
            current_coverage = self.C[:, 0]
            if self.use_sparse:
                point_total_contribution = batched_diffs_sparse(
                    self.K,
                    current_coverage,
                    0,
                    self.num_of_classes,
                    diff_method="abs_diff",
                )
            else:
                point_total_contribution = batched_diffs(
                    self.K,
                    current_coverage,
                    0,
                    self.num_of_classes,
                    diff_method="abs_diff",
                )

            point_total_contribution[curr_l_set] = -np.inf

            sampled_point = torch.argmax(point_total_contribution).item()
            # Do not read the label of sampled_point here: it is still unlabeled.
            kernel_row = self._get_dense_kernel_row(sampled_point)
            self._update_class_agnostic_coverage(kernel_row)

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del kernel_row

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        # Coverage is intentionally not persisted between episodes. It is rebuilt
        # from the current labeled set so baseline and NPC use identical state logic.
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')

        del self.K
        del self.C

        return activeSet, remainSet

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                         checkpoint_file=None):
        """
        Standard interface for active learning sampling.
        Adapts the algorithm to work with the unified AL interface.

        Note: checkpoint_file is ignored for non-DCoM algorithms.
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"

        return self.select_samples(lSet, uSet, trainDataset=trainDataset, checkpoint_file=checkpoint_file)


def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":

            K_batched = K[start:end]
            K_batched = K_batched.to(C.device)
            diff = K_batched.sub(C)
            diff.clamp_(min=0)
            result[start:end] = diff.sum(dim=1)
            del diff
            # result[start:end] = torch.sum(torch.maximum(K_batched - C, torch.zeros_like(K_batched).to(device=C.device)),
            #                               dim=1)
        elif diff_method == "max":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(
                torch.maximum(((K_batched + alpha) / (
                    torch.maximum(K_batched + alpha * number_of_classes, torch.full_like(K_batched, 1e-8)))) - C,
                              torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C,
                              torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result


def batched_diffs_sparse(K, C, alpha=None, number_of_classes=None, chunk_size=1024, diff_method="abs_diff"):
    """
    Computes batched diffs for a Sparse CSR Tensor K and a dense tensor/scalar C.

    Assumes:
    1. K is a torch.sparse_csr_tensor.
    2. C >= 0 (This allows us to ignore the zero-entries in K, keeping operations sparse).
    """
    D, N = K.shape
    device = C.device

    # Extract CSR components and move to device
    crow = K.crow_indices().to(device)  # shape (D+1,)
    ccol = K.col_indices().to(device)  # shape (nnz,)
    cvals = K.values().to(device)  # shape (nnz,)

    # Pre-allocate result on the correct device
    result = torch.zeros(D, device=device, dtype=cvals.dtype)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]  # shape (b,)
        ends = crow[row_start + 1: row_end + 1]  # shape (b,)
        lengths = (ends - starts).to(torch.long)  # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

        # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]  # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(
            torch.arange(b, device=device, dtype=torch.long),
            lengths
        )  # (total_nnz,)

        if diff_method == "abs_diff":
            # --- SPARSE OPTIMIZATION EXPLAINED ---
            # The operation is sum(max(K - C, 0)).
            # For sparse entries where K=0, this is max(0 - C, 0).
            # If C >= 0, this result is 0.
            # Therefore, we only need to work with the non-zero values.

            # Handle broadcasting for C
            if C.numel() == 1:
                # C is a scalar
                c_mapped = C
            elif C.ndim == 1 and C.shape[0] == N:
                # C is a vector matching the columns (feature dim).
                # We gather only the C values that align with the non-zero columns.
                c_mapped = C[cols_all]
            else:
                raise NotImplementedError("C shape not supported for sparse optimization")

            # Perform ReLU(Values - C)
            new_vals = torch.relu(vals_all - c_mapped)

            # Aggregate per row via scatter_add
            chunk_result = torch.zeros(b, device=device, dtype=cvals.dtype)
            chunk_result.scatter_add_(0, row_indices, new_vals)

            result[row_start:row_end] = chunk_result

    return result
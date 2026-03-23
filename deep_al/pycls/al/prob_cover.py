import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import deep_al.pycls.datasets.utils as ds_utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from deep_al.tools.utils import visualize_tsne

class ProbCover:
    def __init__(self, cfg, delta):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        self.all_features = ds_utils.load_features(self.ds_name, self.seed, features_tag=features_tag)

        if delta < 0:
            print("Delta not specified. Starting automated selection process...")
    
            # Track time for performance monitoring
            start_time = time.time()

            print("Running Delta selection algorithm...")

            delta = ds_utils.select_delta(
                features=self.all_features,
                num_classes=self.cfg.MODEL.NUM_CLASSES,
                seed=self.seed
            )

            elapsed = time.time() - start_time
            print(f"Delta selection complete | Value: {delta:.4f} | Time: {elapsed:.2f}s")

        self.delta = delta
        # self.lSet = lSet
        # self.uSet = uSet
        # self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        # self.rel_features = self.all_features[self.relevant_indices]
        # self.graph_df = self.construct_graph()
        # self.activeSet = []
        # print(self.lSet)

    def init_sampling_loop(self,lset, uset):
        self.set_rel_features(lset, uset)
        self.graph_df = self.construct_graph()
        self.activeSet = []

    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        # print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self, lset, uset, num_samples):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {num_samples} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in tqdm(range(num_samples)):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.argsort(degrees, kind='stable')[::-1][0]  # stable sort to get the lowest index in case of ties
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == num_samples, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        return activeSet, remainSet

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None, num_samples=None):
        """
        Standard interface for active learning sampling.

        num_samples: number of samples to select this round (computed from budget / annotator cost).
        Note: checkpoint_file is ignored for non-DCoM algorithms.
        """
        assert num_samples is not None and num_samples > 0, "Expected a positive num_samples"
        assert num_samples < len(uSet), \
            f"num_samples cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and num_samples: {num_samples}"

        return self.select_samples(lSet, uSet, num_samples)

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='ProbCover')
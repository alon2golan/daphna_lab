import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import deep_al.pycls.datasets.utils as ds_utils
from tqdm import tqdm

class DCoM:
    """
    AL algorithm that selects the next centroids based on point density and model confidence, which measured
    using minimum margin.
    """

    def __init__(self, cfg, lSet, uSet, budgetSize, lSet_deltas, max_delta, data_obj=None, plotting_manager=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        features_tag = self.cfg['DATASET'].get('FEATURES_TAG', None)
        self.all_features = ds_utils.load_features(self.ds_name, self.seed, features_tag=features_tag)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.max_delta = max_delta
        self.data_obj = data_obj
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)  # indices of lSet and then uSet in all_features
        # self.relevant_indices = np.arange(self.lSet.size +self.uSet.size).astype(int)  # indices of lSet and then uSet in all_features
        self.rel_features = self.all_features[self.relevant_indices]  # features of lSet and then uSet
        
        if self.cfg.ACTIVE_LEARNING.INITIAL_DELTA < 0:
            print("Delta not specified. Starting automated selection process...")
    
            # Track time for performance monitoring
            start_time = time.time()

            print("Running Delta selection algorithm...")

            self.cfg.ACTIVE_LEARNING.INITIAL_DELTA = ds_utils.select_delta(
                features=self.all_features,
                num_classes=self.cfg.MODEL.NUM_CLASSES,
                classifier_on_raw_images=(self.cfg.EVAL_MODEL_TYPE == "from_images")
            )

            elapsed = time.time() - start_time
            print(f"Delta selection complete | Value: {self.cfg.ACTIVE_LEARNING.INITIAL_DELTA:.4f} | Time: {elapsed:.2f}s")

        # Initialize delta list - if empty (first iteration), create initial deltas
        if len(lSet_deltas) == 0:
            print('DCoM: Initializing delta list with initial_delta for first budget')
            self.lSet_deltas = [float(self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * budgetSize
        else:
            self.lSet_deltas = [float(delta) for delta in lSet_deltas]
        
        self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
        self.delta_avg = np.average(self.lSet_deltas) if self.lSet_deltas else self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
        
        self.plotting_manager = plotting_manager
        # Initialize algorithm state if plotting_manager is provided
        if plotting_manager is not None:
            self.plotting_manager.add_delta_stats(
                self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, 0.0
            )

    def construct_graph_excluding_lSet(self, delta=None, batch_size=700):
        """
        Creates a directed graph where:
        x -> y if l2(x, y) < delta.
        Considers all images, but does not reference or delete edges in lSet.

        The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
        """
        if delta is None:
            delta = self.delta_avg

        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Before delete lSet neighbors: Graph contains {len(df)} edges.')
        return df

    def construct_graph(self, delta=None, batch_size=500):
        """
         Creates a directed graph where:
         x -> y if l2(x, y) < delta, and deletes the covered points using lSet_deltas.

         Deletes edges to the covered samples (samples that are covered by lSet balls)
         and deletes all the edges from lSet.

         The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
         """
        if delta is None:
            delta = self.delta_avg

        df = self.construct_graph_excluding_lSet(delta, batch_size)

        # removing incoming edges to all cover from the existing labeled set
        edges_from_lSet_in_ball = np.isin(df.x, np.arange(len(self.lSet))) & (df.d < df.x.map(self.lSet_deltas_dict))
        # edges_from_lSet_in_ball = np.isin(df.x, self.lSet) & (df.d < df.x.map(self.lSet_deltas_dict))
        covered_samples = df.y[edges_from_lSet_in_ball].unique()

        edges_to_covered_samples = np.isin(df.y, covered_samples)
        all_edges_from_lSet = np.isin(df.x, np.arange(len(self.lSet)))
        # all_edges_from_lSet = np.isin(df.x, self.lSet)

        mask = all_edges_from_lSet | edges_to_covered_samples  # all the points inside the balls
        df_filtered = df[~mask]

        print(f'Finished constructing graph using delta={delta}')
        print(f'Graph contains {len(df_filtered)} edges.')
        return df_filtered, covered_samples

    def select_samples(self, model, train_data, data_obj):
        """
        Performs DCoM active sampling section.
        """
        def get_competence_score(coverage):
            """
            Implementation of the logistic function weighting.
            """
            k = self.cfg.ACTIVE_LEARNING.K_LOGISTIC  # the logistic growth rate or steepness of the curve
            a = self.cfg.ACTIVE_LEARNING.A_LOGISTIC  # the logistic function center
            p = (1 + np.exp(-k * (1 - a)))
            competence_score = p / (1 + np.exp(-k * (coverage - a)))
            print(f'a = {a}, k = {k}, p = {round(p, 4)}')
            print("The coverage over the graph is: ", coverage)
            print("The competence_score is: ", round(competence_score, 3))
            return round(competence_score, 3)

        print(f"\n==================== Start DCoM Active Sampling ====================")
        # Calculate the current coverage
        selected = []
        fully_graph = self.construct_graph_excluding_lSet(self.max_delta)
        current_coverage = DCoM.calculate_coverage(fully_graph, self.lSet, self.lSet_deltas_dict, len(self.relevant_indices))
        del fully_graph

        competence_score = get_competence_score(current_coverage)
        # competence_score = 0

        margin = self.calculate_margin(model, train_data, data_obj)
        margin[0: len(self.lSet)] = 0 # We define the margin score to be 1-margin (as described in our paper)
        # if self.lSet.size>0:
        #     margin[self.lSet.astype(int)] = 0 # We define the margin score to be 1-margin (as described in our paper)

        cur_df, covered_samples = self.construct_graph(self.delta_avg)

        for i in range(self.budgetSize): # The active selection
            coverage = len(covered_samples) / len(self.relevant_indices)

            if len(cur_df) == 0:
                ranks = np.zeros(len(self.relevant_indices))
            else:
                # calculate density for each point
                ranks = self.calculate_density(cur_df)
            # cur_chose = np.concatenate((self.lSet.astype(int), selected)).astype(int)
            # if cur_chose.size > 0:
            #     ranks[cur_chose] = -1
            cur_selection = DCoM.normalize_and_maximize(ranks, margin, 1, lambda r, e: competence_score * e + (1 - competence_score) * r)[0]
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tCoverage is {coverage:.3f}. \tCurr choice is {cur_selection}. \tcompetence_score={competence_score}')

            new_covered_samples = cur_df.y[(cur_df.x == cur_selection)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'

            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]  # Delete all the edges to the covered samples
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            margin[cur_selection] = 0
            selected.append(cur_selection)

        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        assert len(np.intersect1d(self.lSet, activeSet)) == 0, 'all samples should be new'
        print(f'Finished the selection of {len(activeSet)} samples.')
        return activeSet, remainSet

    def new_centroids_deltas(self, lSet_labels, pseudo_labels, budget, batch_size=500, all_labels=[]):
        """
        Performs binary search of the next delta values.
        """
        def calc_threshold(coverage):
            assert 0 <= coverage <= 1, f'coverage is not between 0 to 1: {coverage}'
            return 0.2 * coverage + 0.4

        def check_purity(df, cent_label, delta):
            # find all the neighbors
            edges_from_lSet = (df.x == cent_idx) & (df.d < delta)
            neighbors_idx = df.y[edges_from_lSet]

            # take their neighbors and compute the ball purity (Are the labels the same as the chosen point?)
            neighbors_pseudo_labels = list(map(pseudo_labels.__getitem__, neighbors_idx))
            neighbors_real_labels = list(map(all_labels.__getitem__, neighbors_idx))

            if len(neighbors_idx):
                pseudo_purity = sum(np.array(neighbors_pseudo_labels == cent_label)) / len(neighbors_idx)
                real_purity = sum(np.array(neighbors_real_labels == cent_label)) / len(neighbors_idx)
                print(f'real_purity: {real_purity}, pseudo_purity: {pseudo_purity}')
                return pseudo_purity
            return 0

        new_deltas = []
        df = self.construct_graph_excluding_lSet(self.max_delta, batch_size)

        fully_df = self.construct_graph_excluding_lSet(self.max_delta)
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(self.lSet))) & (
        # covered_samples = fully_df.y[np.isin(fully_df.x, self.lSet) & (
                fully_df.d < fully_df.x.map(self.lSet_deltas_dict))].unique()
        coverage = len(covered_samples) / len(self.relevant_indices)

        purity_threshold = calc_threshold(coverage)
        print("Current threshold: ", purity_threshold)

        for cent_idx, centroid in enumerate(self.lSet):
            if cent_idx < len(self.lSet) - budget:  # Not new points
                continue

            print(f'start calculation for cent_idx: {cent_idx}')
            low_del_val = 0
            max_del_val = self.max_delta
            mid_del_val = (low_del_val + max_del_val) / 2
            last_purity = 0
            last_delta = mid_del_val

            while abs(low_del_val - max_del_val) > self.cfg.ACTIVE_LEARNING.DELTA_RESOLUTION:
                curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
                print("centroid: ", centroid, ", idx: ", cent_idx, ". delta = ", mid_del_val, " and purity = ", curr_purity)

                if last_delta < mid_del_val and last_purity == purity_threshold and curr_purity < purity_threshold:
                    mid_del_val = last_delta
                    break

                if curr_purity < purity_threshold:
                    # if smaller than the threshold - try smaller delta
                    max_del_val = mid_del_val
                elif curr_purity >= purity_threshold:  # if bigger than threshold -> try bigger delta
                    low_del_val = mid_del_val

                last_purity = curr_purity
                last_delta = mid_del_val
                mid_del_val = (low_del_val + max_del_val) / 2

            curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
            print("the chosen delta: ", mid_del_val, "and its purity: ", curr_purity)
            print("---------------------------------------------------------------------------")
            new_deltas.append(str(round(mid_del_val, 2)))

        self.lSet_deltas = [float(delta) for delta in new_deltas]
        self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
        print("All new deltas: ", new_deltas, '\n')
        return new_deltas

    def calculate_density(self, df):
        rank_mapping = pd.DataFrame(df.groupby('x')['y'].count())
        all_indices_df = pd.DataFrame(index=np.arange(len(self.relevant_indices)))
        result_df = pd.merge(all_indices_df, rank_mapping, left_index=True, right_index=True, how='left').fillna(0)
        return np.array(result_df)

    def calculate_margin(self, model, train_data, data_obj):
        oldmode = model.training
        model.eval()

        print(f'Start calculating points margin.')
        all_images_idx = self.relevant_indices
        images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                batch_size=self.cfg.TRAIN.BATCH_SIZE, data=train_data)
        clf = model.cuda()
        ranks = []
        n_loader = len(images_loader)

        for i, (x_u, _) in enumerate(tqdm(images_loader, desc="All images Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]

                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
                difference = -1 * difference
                ranks.append(difference.detach().cpu().numpy())
        ranks = np.concatenate(ranks, axis=0)
        print(f"u_ranks.shape: {ranks.shape}")

        model.train(oldmode)

        margin_result = np.array(ranks).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_margin_result = scaler.fit_transform(margin_result)
        final_margin_result = np.array(normalized_margin_result.flatten().tolist())
        return final_margin_result

    @staticmethod
    def normalize_and_maximize(param1_list, param2_list, amount, target_func):
        """
        Perform pre-processing on each list and apply the target function on them.
        """
        param1_arr = np.array(param1_list).reshape(-1, 1)
        param2_arr = np.array(param2_list).reshape(-1, 1)

        # Min-Max normalization using scikit-learn's MinMaxScaler
        scaler = MinMaxScaler()
        param1_normalized = scaler.fit_transform(param1_arr)
        param2_normalized = scaler.fit_transform(param2_arr)

        # Calculate the product using the provided target_func
        product_array = target_func(param1_normalized.flatten(), param2_normalized.flatten())

        sorted_indices = np.argsort(product_array, kind='stable')[::-1]

        return sorted_indices[:amount]

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None):
        """
        Complete DCoM iteration: update state → sample → prepare next.
        
        Handles all state management internally:
        1. Update deltas from checkpoint if available
        2. Perform sampling
        3. Prepare deltas for next iteration and update cfg.ACTIVE_LEARNING.DELTA_LST
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
        
        # Update sets in the instance (in case they changed)
        self.lSet = lSet
        self.uSet = uSet
        
        # CRITICAL: Recompute derived state that depends on lSet/uSet
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        
        # Stage 1: Update state from previous episode (if checkpoint provided)
        if checkpoint_file is not None and self.data_obj is not None:
            self._update_state_from_checkpoint(checkpoint_file, lSet, trainDataset)
        
        # Stage 2: Perform sampling
        activeSet, remainSet = self.select_samples(clf_model, trainDataset, self.data_obj)
        
        # Stage 3: Prepare state for next iteration and update global delta list
        if checkpoint_file is not None:
            self._prepare_state_for_next_iteration()
        
        return activeSet, remainSet

    def _update_state_from_checkpoint(self, checkpoint_file, lSet, train_data):
        """
        Update algorithm state based on trained model checkpoint.
        
        Computes and updates delta values dynamically based on model predictions.
        """
        print("======== Update the deltas dynamically ========\n")
        
        from deep_al.pycls.utils.model_handler import get_label_from_model
        
        # Prepare labels and data loaders
        lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
        all_images_idx = np.array(list(lSet) + list(self.uSet))
        images_loader = self.data_obj.getSequentialDataLoader(
            indexes=all_images_idx,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            data=train_data
        )
        all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))
        
        # Get pseudo-labels from the trained model
        images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, self.cfg)
        
        # Update deltas based on model performance
        new_deltas = self.new_centroids_deltas(
            lSet_labels,
            all_labels=all_labels,
            pseudo_labels=images_pseudo_labels,
            budget=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
        )
        
        # Update delta history and values
        self.lSet_deltas = [float(delta) for delta in new_deltas]
        self.lSet_deltas_dict = dict(zip(np.arange(len(self.lSet)), self.lSet_deltas))
        self.delta_avg = np.average(self.lSet_deltas)
        
        # Track statistics for visualization (if plotting_manager available)
        delta_lst_float = [float(delta) for delta in new_deltas]
        self.last_delta_avg = np.average(delta_lst_float)
        self.last_delta_std = np.std(delta_lst_float)
        if self.plotting_manager is not None:
            self.plotting_manager.add_delta_stats(self.last_delta_avg, self.last_delta_std)

    def _prepare_state_for_next_iteration(self):
        """
        Prepare algorithm state for the next iteration.
        
        Computes the average of current deltas and initializes deltas for
        the next batch of points to be selected.
        """
        delta_lst_float = [float(delta) for delta in self.lSet_deltas]
        avg_delta = np.average(delta_lst_float)
        next_deltas = [avg_delta] * self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
        
        # Extend the delta list for next iteration
        self.lSet_deltas = self.lSet_deltas + next_deltas
        
        print(f"Current delta list length: {len(self.lSet_deltas)}")
        if hasattr(self, 'last_delta_avg'):
            print("Current delta avg: ", self.last_delta_avg)
            print("Current delta std: ", self.last_delta_std)

    @staticmethod
    def calculate_coverage(fully_df, lSet, lSet_deltas_dict, total_data_len):
        """
        Return the current probability coverage.
        """
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(lSet))) & (
                fully_df.d < fully_df.x.map(lSet_deltas_dict))].unique()  # lSet send arrow to them
        return len(covered_samples) / total_data_len



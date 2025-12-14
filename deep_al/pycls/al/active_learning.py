# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import numpy as np
from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
from .MISP import MISP
from .MAX_MISP import MAX_MISP
from .ALL_MISP import ALL_MISP
from .BAYES_MISP import BAYES_MISP
from .prob_cover import ProbCover
from .MISP_probcover import MISPC
from .MISP_plus import MISP_PLUS
from .typiclust import TypiClust
from .maxherding import MaxHerding
from .DCoM import DCoM
import deep_al.pycls.utils.logging as lu

logger = lu.get_logger(__name__)


class ActiveLearning:
    """
    Active Learning orchestrator that delegates to concrete AL strategies.
    
    Uses a factory pattern to instantiate the appropriate algorithm based on config,
    then delegates sampling to that algorithm's sample_from_uSet() method.
    """

    def __init__(self, dataObj, cfg, train_labels=None, lset=None):
        self.dataObj = dataObj
        self.cfg = cfg
        self.train_labels = train_labels
        self.lset = lset
        
        # Create the appropriate algorithm instance using factory
        self.al_algorithm = self._create_algorithm(train_labels, lset)
        
        # Initialize algorithm-specific instance if needed (for DCoM)
        self.al_algorithm_instance = None
        self.algorithm_state = {}  # Store algorithm-specific dynamic state
        self._initialize_algorithm_instance(lset)

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using the configured algorithm.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]
        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
        
        logger.info(f"Using {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling function for active learning.")
        
        # Delegate to the algorithm instance
        activeSet = self.al_algorithm.sample_from_uSet(
            clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels
        )
        
        # Calculate remaining unlabeled set
        new_uSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        
        return activeSet, new_uSet
        

    def _create_algorithm(self, train_labels=None, lset=None):
        """
        Factory method to create the appropriate AL algorithm instance.
        Returns the algorithm object with sample_from_uSet() method.
        """
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        # MISP variants
        if sampling_fn == "misp":
            return MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                       delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        elif sampling_fn == "max_misp":
            return MAX_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                           delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        elif sampling_fn == "all_misp":
            return ALL_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                           train_labels=train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
        
        elif sampling_fn == "bayes_misp":
            return BAYES_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                             train_labels=train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
        
        # ProbCover variants
        elif sampling_fn in ["prob_cover", "probcover"]:
            return ProbCover(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        elif sampling_fn == "mispc":
            return MISPC(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        # Dynamic algorithms (directly instantiate with lSet and uSet at sampling time)
        elif sampling_fn == "misp_plus":
            return MISP_PLUS(self.cfg, lset if lset is not None else [], uSet=np.array([]),
                            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            train_data=None, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        elif sampling_fn.startswith("typiclust"):
            is_scan = sampling_fn.endswith('dc')
            return TypiClust(self.cfg, lset if lset is not None else [], np.array([]),
                            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
        
        elif sampling_fn in ["maxherding", "max_herding"]:
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            return MaxHerding(self.cfg, lset if lset is not None else [], np.array([]),
                            self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, delta=delta)
        
        # DCoM (handled separately for state management)
        elif sampling_fn == "dcom":
            delta_lst = self.cfg.ACTIVE_LEARNING.DELTA_LST if hasattr(self.cfg.ACTIVE_LEARNING, 'DELTA_LST') else []
            return DCoM(self.cfg, lset if lset is not None else [], [],
                       budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                       max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                       lSet_deltas=delta_lst)
        
        # Traditional sampling methods and special cases (from Sampling class)
        elif sampling_fn in ["random", "uncertainty", "entropy", "margin", "dbal", "bald", "ensemble_var_r"]:
            return Sampling(dataObj=self.dataObj, cfg=self.cfg)
        elif sampling_fn == "coreset":
            return CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
        elif sampling_fn == "vaal":
            return AdversarySampler(dataObj=self.dataObj, cfg=self.cfg)
        
        else:
            raise NotImplementedError(f"{sampling_fn} is either not implemented or there is some spelling mistake.")

    def _initialize_algorithm_instance(self, lset):
        """Initialize algorithm-specific instance (DCoM, etc.)."""
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        if sampling_fn == "dcom":
            # Initialize delta list from config or create new
            delta_lst = self.cfg.ACTIVE_LEARNING.DELTA_LST if hasattr(self.cfg.ACTIVE_LEARNING, 'DELTA_LST') else []
            
            self.al_algorithm_instance = DCoM(
                self.cfg, lset if lset is not None else [], [],
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                lSet_deltas=delta_lst
            )
            
            # Store dynamic state in algorithm_state
            if lset is not None and len(lset) == 0:
                delta_lst = [str(self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
            
            self.algorithm_state['delta_lst'] = delta_lst
    
    def initialize_algorithm_state(self, plotting_manager):
        """Initialize algorithm-specific state on first iteration."""
        if self.al_algorithm_instance is None:
            return
        
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        if sampling_fn == "dcom":
            # Initialize delta stats for DCoM
            plotting_manager.add_delta_stats(
                self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, 0.0
            )
    
    def update_algorithm_state(self, data_obj, checkpoint_file, lSet, train_data, uSet, plotting_manager):
        """Update algorithm-specific state after each episode (DCoM deltas, etc.)."""
        if self.al_algorithm_instance is None:
            return
        
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        if sampling_fn == "dcom":
            print("======== Update the deltas dynamically ========\n")
            
            from deep_al.pycls.utils.model_handler import get_label_from_model
            
            lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
            all_images_idx = np.array(list(lSet) + list(uSet))
            images_loader = data_obj.getSequentialDataLoader(
                indexes=all_images_idx,
                batch_size=self.cfg.TRAIN.BATCH_SIZE,
                data=train_data
            )
            all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))
            
            images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, self.cfg)
            
            # Update deltas
            new_deltas = self.al_algorithm_instance.new_centroids_deltas(
                lSet_labels,
                all_labels=all_labels,
                pseudo_labels=images_pseudo_labels,
                budget=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
            )
            
            # Update dynamic state
            delta_lst = self.algorithm_state['delta_lst']
            delta_lst[-1 * self.cfg.ACTIVE_LEARNING.BUDGET_SIZE:] = new_deltas
            self.algorithm_state['delta_lst'] = delta_lst
            
            delta_lst_float = [float(delta) for delta in delta_lst]
            plotting_manager.add_delta_stats(np.average(delta_lst_float), np.std(delta_lst_float))
    
    def prepare_next_iteration(self):
        """Prepare algorithm-specific state for next iteration."""
        if self.al_algorithm_instance is None:
            return
        
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        if sampling_fn == "dcom":
            delta_lst = self.algorithm_state['delta_lst']
            delta_lst_float = [float(delta) for delta in delta_lst]
            next_deltas = [str(round(np.average(delta_lst_float), 2))] * self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
            delta_lst.extend(next_deltas)
            self.algorithm_state['delta_lst'] = delta_lst
            return delta_lst_float
    
    def has_algorithm_instance(self):
        """Check if algorithm requires state management."""
        return self.al_algorithm_instance is not None


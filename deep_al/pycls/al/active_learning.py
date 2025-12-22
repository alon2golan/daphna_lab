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
    Initializes algorithm-specific state (e.g., DCoM delta tracking) automatically.
    """

    def __init__(self, dataObj, cfg, train_labels=None, lset=None, plotting_manager=None):
        self.cfg = cfg
        
        # Create the appropriate algorithm instance using factory
        self.al_algorithm = self._create_algorithm(train_labels, lset, dataObj, plotting_manager)

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None):
        """
        Sample from uSet using the configured algorithm.
        
        Delegates to algorithm instance which has access to data_obj and plotting_manager
        stored in its constructor.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]
        lSet, uSet: Labeled and unlabeled set indices
        trainDataset: Training dataset object
        supportingModels: List of models which are used for sampling process.
        checkpoint_file: Model checkpoint from training (for DCoM state updates only)

        OUTPUT
        -------
        Returns activeSet, new_uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), \
            f"BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and budgetSize: {self.cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
        
        logger.info(f"Using {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling function for active learning.")
        
        # Delegate to the algorithm instance
        # Algorithms have data_obj and plotting_manager stored in constructor
        activeSet = self.al_algorithm.sample_from_uSet(
            clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
            checkpoint_file=checkpoint_file
        )
        
        # Calculate remaining unlabeled set
        new_uSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        
        return activeSet, new_uSet
        

    def _create_algorithm(self, train_labels=None, lset=None, dataObj=None, plotting_manager=None):
        """
        Factory method to create the appropriate AL algorithm instance.
        Passes data_obj and plotting_manager to each algorithm.
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
            return MISPC(self.cfg, lset if lset is not None else [], np.array([]),
                        budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        # Dynamic algorithms
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
                       lSet_deltas=delta_lst,
                       data_obj=dataObj, plotting_manager=plotting_manager)
        
        # Traditional sampling methods and special cases (from Sampling class)
        elif sampling_fn in ["random", "uncertainty", "entropy", "margin", "dbal", "bald", "ensemble_var_r"]:
            return Sampling(dataObj=dataObj, cfg=self.cfg)
        elif sampling_fn == "coreset":
            return CoreSetMIPSampling(cfg=self.cfg, dataObj=dataObj)
        elif sampling_fn == "vaal":
            return AdversarySampler(dataObj=dataObj, cfg=self.cfg)
        
        else:
            raise NotImplementedError(f"{sampling_fn} is either not implemented or there is some spelling mistake.")


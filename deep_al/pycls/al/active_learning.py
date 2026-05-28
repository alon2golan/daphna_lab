# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import time
import numpy as np
from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler

from .prob_cover import ProbCover
from .typiclust import TypiClust
from .maxherding import MaxHerding
from .DCoM import DCoM
from .coverage_matrix_methods import CoverageMatrixMethod
import deep_al.pycls.utils.logging as lu
import deep_al.pycls.datasets.utils as ds_utils

logger = lu.get_logger(__name__)


# AL methods that operate purely on pre-computed features and never call any
# method on clf_model. These are the only AL methods that work when clf_model
# is a sklearn KNeighborsClassifier (i.e. cfg.MODEL.USE_1NN=True). Every other
# method calls nn.Module-only APIs (.eval(), .train(), .penultimate_active,
# forward passes for logits / dropout) and would raise AttributeError under 1-NN.
NN1_COMPATIBLE_METHODS = frozenset({
    'random',
    'probcover', 'prob_cover', 'probcover_orig', 'prob_cover_orig',
    'maxherding', 'max_herding', 'maxherding_orig', 'max_herding_orig',
})


def _is_nn1_compatible(sampling_fn):
    fn = sampling_fn.lower()
    # typiclust dispatch in _create_algorithm uses startswith(), so match the same way
    if fn.startswith('typiclust'):
        return True
    return fn in NN1_COMPATIBLE_METHODS


def validate_method_eval_compat(cfg):
    """Reject AL method x eval_model_type combinations that can only fail.

    Raises ValueError when USE_1NN is True but the chosen AL method needs a
    parametric classifier (logits, gradients, or penultimate features). Catches
    the failure at config-load time instead of after dataset loading, with a
    useful message instead of a downstream AttributeError.
    """
    if not cfg.MODEL.USE_1NN:
        return
    if _is_nn1_compatible(cfg.ACTIVE_LEARNING.SAMPLING_FN):
        return
    raise ValueError(
        "AL method '{fn}' is incompatible with --eval_model_type NN1/NN1_normalized: "
        "it requires a parametric model (logits, gradients, or penultimate features), "
        "but USE_1NN is True so clf_model is a sklearn KNeighborsClassifier. "
        "Use --eval_model_type from_features instead, or pick one of: {ok}.".format(
            fn=cfg.ACTIVE_LEARNING.SAMPLING_FN,
            ok=sorted(NN1_COMPATIBLE_METHODS | {'typiclust*'}),
        )
    )


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
        activeSet, new_uSet = self.al_algorithm.sample_from_uSet(
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
        
        if sampling_fn in ["prob_cover_orig", "probcover_orig"]:
            return ProbCover(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
        elif sampling_fn in ["prob_cover", "probcover"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'tophat':
                logger.info('Warning!! for ProbCover the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                print('Warning!! for ProbCover the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')

                logger.info('Warning!! Running ProbCover as Sparse. Changing K_sparse to True')
                print('Warning!! Running ProbCover as Sparse. Changing K_sparse to True')

                self.cfg['KERNEL_TYPE'] = 'tophat'
                self.cfg['K_SPARSE'] = True
            cmm = CoverageMatrixMethod(self.cfg, train_labels=train_labels)
            return cmm
                
        elif sampling_fn.startswith("typiclust"):
            is_scan = sampling_fn.endswith('dc')
            return TypiClust(self.cfg, lset if lset is not None else [], np.array([]),
                            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
        elif sampling_fn in ["maxherding_orig", "max_herding_orig"]:
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            return MaxHerding(self.cfg, lset if lset is not None else [], np.array([]),
                            self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, delta=delta)
        
        elif sampling_fn in ["maxherding", "max_herding"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'rbf':
                logger.info('Warning!! for MaxHerding the Kernel must be RBF. Changing KERNEL_TYPE to rbf')
                print('Warning!! for MaxHerding the Kernel must be TopHat. Changing KERNEL_TYPE to rbf')
                self.cfg['KERNEL_TYPE'] = 'rbf'
            cmm = CoverageMatrixMethod(self.cfg, train_labels=train_labels)
            return cmm

        # DCoM (handles its own delta initialization internally)
        elif sampling_fn == "dcom":
            return DCoM(self.cfg, lset if lset is not None else [], [],
                       budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                       max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                       lSet_deltas=[],  # DCoM will initialize if empty
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


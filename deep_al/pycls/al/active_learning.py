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


class ActiveLearning:
    """
    Active Learning orchestrator that delegates to concrete AL strategies.
    
    Uses a factory pattern to instantiate the appropriate algorithm based on config,
    then delegates sampling to that algorithm's sample_from_uSet() method.
    Initializes algorithm-specific state (e.g., DCoM delta tracking) automatically.
    """

    def __init__(self, dataObj, cfg, train_labels=None, lset=None, plotting_manager=None, annotator_manager=None):
        self.cfg = cfg
        self.annotator_manager = annotator_manager
        # Create the appropriate algorithm instance using factory
        self.al_algorithm = self._create_algorithm(train_labels, lset, dataObj, plotting_manager)

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None, num_samples=None):
        """
        Sample from uSet using the configured algorithm.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]
        lSet, uSet: Labeled and unlabeled set indices
        trainDataset: Training dataset object
        supportingModels: List of models which are used for sampling process.
        checkpoint_file: Model checkpoint from training (for DCoM state updates only)
        num_samples: Number of samples to select from uSet this round. This is
                     computed by the caller from the annotator budget and cost-per-sample,
                     so cheap annotators can afford more samples than expensive ones.

        OUTPUT
        -------
        Returns activeSet, new_uSet
        """

        assert num_samples > 0, "Expected a positive num_samples"
        assert num_samples < len(uSet), \
            f"num_samples cannot exceed length of unlabelled set. Length of unlabelled set: {len(uSet)} and num_samples: {num_samples}"

        logger.info(f"Using {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling function for active learning.")

        activeSet, remain_set = self.al_algorithm.sample_from_uSet(
            clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
            checkpoint_file=checkpoint_file, num_samples=num_samples
        )

        return activeSet, remain_set
        

    def _create_algorithm(self, train_labels=None, lset=None, dataObj=None, plotting_manager=None):
        """
        Factory method to create the appropriate AL algorithm instance.
        Passes data_obj and plotting_manager to each algorithm.
        """
        sampling_fn = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
        
        if sampling_fn in ["prob_cover_orig", "probcover_orig"]:
            return ProbCover(self.cfg, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        
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
                            is_scan=is_scan)
        elif sampling_fn in ["maxherding_orig", "max_herding_orig"]:
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            return MaxHerding(self.cfg, lset if lset is not None else [], np.array([]),
                            delta=delta)
        
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
                       max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                       lSet_deltas=[],
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


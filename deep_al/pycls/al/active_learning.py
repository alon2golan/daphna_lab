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

        elif sampling_fn in ["prob_cover", "probcover", "npc_probcover_crossval", "npc_probcover_knn",
                             "npc_probcover_aum", "npc_probcover_oracle"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'tophat':
                logger.info('Warning!! for ProbCover/NPC the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                print('Warning!! for ProbCover/NPC the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')

                logger.info('Warning!! Running ProbCover/NPC as Sparse. Changing SPARSE_K to True')
                print('Warning!! Running ProbCover/NPC as Sparse. Changing SPARSE_K to True')

                self.cfg['KERNEL_TYPE'] = 'tophat'
                self.cfg['SPARSE_K'] = True

            if sampling_fn == "npc_probcover_crossval":
                self.cfg.ACTIVE_LEARNING.NAS_FILTER_METHOD = "crossval"
            elif sampling_fn == "npc_probcover_knn":
                self.cfg.ACTIVE_LEARNING.NAS_FILTER_METHOD = "knn"
            elif sampling_fn == "npc_probcover_aum":
                self.cfg.ACTIVE_LEARNING.NAS_FILTER_METHOD = "aum"
            elif sampling_fn == "npc_probcover_oracle":
                self.cfg.ACTIVE_LEARNING.NAS_FILTER_METHOD = "oracle"

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

        elif sampling_fn in [
            "maxherding", "max_herding",
            "npc_maxherding_crossval", "npc_maxherding_knn",
            "npc_maxherding_aum", "npc_maxherding_oracle",
        ]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'rbf':
                logger.info('Warning!! for MaxHerding/NPC-MaxHerding the kernel must be RBF. Changing KERNEL_TYPE to rbf')
                print('Warning!! for MaxHerding/NPC-MaxHerding the kernel must be RBF. Changing KERNEL_TYPE to rbf')
                self.cfg['KERNEL_TYPE'] = 'rbf'

            if sampling_fn.startswith("npc_maxherding_"):
                detector = sampling_fn.removeprefix("npc_maxherding_")
                if detector not in {"crossval", "knn", "aum", "oracle"}:
                    raise ValueError(f"Unsupported NPC-MaxHerding detector: {detector}")
                self.cfg.ACTIVE_LEARNING.NAS_FILTER_METHOD = detector

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

        # MISP family methods - use existing Prob_Misp.ALL_MISP implementation as a fallback
        elif sampling_fn in ["misp", "max_misp", "all_misp", "bayes_misp"]:
            from .Prob_Misp import ALL_MISP
            # ALL_MISP constructor expects train_labels and lset; pass them when available
            return ALL_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            train_labels=train_labels, lset=lset, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)

        else:
            raise NotImplementedError(f"{sampling_fn} is either not implemented or there is some spelling mistake.")


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
from .coverage_matrix_methods import CoverageMatrixMethod, NoiseAwareCoverage
from .bayesian_coverage import BayesianCoverage
from .noise_filters import build_noise_filter
import deep_al.pycls.utils.logging as lu
import deep_al.pycls.datasets.utils as ds_utils

logger = lu.get_logger(__name__)


def _ensure_bayes_misp_cfg_defaults(cfg):
    """Fill in the cfg keys BAYES_MISP reads that our fork doesn't declare.

    Values are Itai's train_al.py CLI defaults (bayes_al_2025@main). Keys that
    already exist (e.g. set via our CLI flags --alpha/--diff_method/--cont_method)
    are left untouched.
    """
    defaults = {
        'K_SPARSITY_THRESHOLD': 0.0,
        'CALC_METHOD': 'max',
        'ALPHA_INIT_MODE': 'constant',
        'ALPHA_VECTOR_PATH': '',
        'ALPHA_LOWER_BOUND': 1.0,
        'ALPHA_UPPER_BOUND': 1.0,
        'ALPHA_DECAY_GAMMA': 0.0,
        'DISTRIBUTION_CONT_WEIGHT_METHOD': 'weighted',
        'C_NORMALIZATION': 'sum',
        'CLASS_WEIGHTING_METHOD': 'none',
        'UPDATE_K_MATRIX': False,
        'SWITCH_ALPHA_LOW_TO_HIGH': False,
        'SWITCH_ALPHA_HIGH_TO_LOW': False,
        'SWITCH_ALPHA_ALLTIME': False,
        'LOCAL_ALPHA': False,
        'LOCAL_ALPHA_ORACLE_METHOD': 'entropy',
    }
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value


class BayesMispAdapter:
    """Adapts Itai's stateful BAYES_MISP (select_samples(lset, uset)) to our
    sample_from_uSet interface.

    The wrapped object owns the C/K posterior state across episodes. Labels it
    consumes during the greedy selection loop come from the observed-label table
    it was constructed with (the annotator's precomputed noisy_labels), so the
    orchestrator's post-hoc annotate_samples() call assigns the exact same
    labels and only settles the cost accounting.

    posterior_engine (optional, MispPosteriorEngine): supplies the design-§5
    training targets (TRAIN_LABEL_MODE != 'observed', e.g. pseudo) that
    BAYES_MISP itself has no notion of. The adapter mirrors the
    BayesianCoverage training-target API (_selection_done / pseudo_scope /
    training_targets) so train_al._bayes_training_loader treats a misp run
    exactly like a bayes_product one; the engine is re-synced from the
    annotated labeled set on every training_targets call.
    """

    def __init__(self, misp, posterior_engine=None):
        self.misp = misp
        self.posterior_engine = posterior_engine
        self._post_selection_sets = None

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                         checkpoint_file=None, num_samples=None, **kwargs):
        if num_samples is not None:
            self.misp.budgetSize = int(num_samples)
        activeSet, remainSet = self.misp.select_samples(
            np.asarray(lSet, dtype=int), np.asarray(uSet, dtype=int))
        if self.posterior_engine is not None:
            lset_after = np.concatenate([np.asarray(lSet, dtype=int),
                                         np.asarray(activeSet, dtype=int)])
            self._post_selection_sets = (lset_after, np.asarray(remainSet, dtype=int))
        return activeSet, remainSet

    # --- BayesianCoverage training-target API (train_al._bayes_training_loader) ---
    @property
    def _selection_done(self):
        return self.posterior_engine is not None and self._post_selection_sets is not None

    @property
    def pseudo_scope(self):
        return self.posterior_engine.pseudo_scope

    def training_targets(self, global_indices, mode=None, pseudo_k=None):
        if self.posterior_engine is None:
            raise RuntimeError(
                "bayes_misp run requested TRAIN_LABEL_MODE != 'observed' but no "
                "posterior engine was constructed — training targets unavailable.")
        lset, uset = self._post_selection_sets
        self.posterior_engine.sync(lset, uset)
        return self.posterior_engine.training_targets(global_indices, mode=mode,
                                                      pseudo_k=pseudo_k)


class ProbCoverPseudoAdapter:
    """ProbCover selection + posterior-engine training targets (the §4.8
    probcover × pseudo control arm). Same shape as BayesMispAdapter: the inner
    CoverageMatrixMethod owns selection, the MispPosteriorEngine (product
    combine by default) manufactures TRAIN_LABEL_MODE != 'observed' targets,
    and this adapter exposes the BayesianCoverage training-target API so
    train_al._bayes_training_loader treats the run like a bayes_product one."""

    def __init__(self, cmm, posterior_engine):
        self.cmm = cmm
        self.posterior_engine = posterior_engine
        self._post_selection_sets = None

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                         checkpoint_file=None, num_samples=None, annotator_id=None, **kwargs):
        activeSet, remainSet = self.cmm.sample_from_uSet(
            clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
            checkpoint_file=checkpoint_file, num_samples=num_samples,
            annotator_id=annotator_id)
        if self.posterior_engine is not None:
            lset_after = np.concatenate([np.asarray(lSet, dtype=int),
                                         np.asarray(activeSet, dtype=int)])
            self._post_selection_sets = (lset_after, np.asarray(remainSet, dtype=int))
        return activeSet, remainSet

    @property
    def _selection_done(self):
        return self.posterior_engine is not None and self._post_selection_sets is not None

    @property
    def pseudo_scope(self):
        return self.posterior_engine.pseudo_scope

    def training_targets(self, global_indices, mode=None, pseudo_k=None):
        lset, uset = self._post_selection_sets
        self.posterior_engine.sync(lset, uset)
        return self.posterior_engine.training_targets(global_indices, mode=mode,
                                                      pseudo_k=pseudo_k)


class ActiveLearning:
    """
    Active Learning orchestrator that delegates to concrete AL strategies.
    
    Uses a factory pattern to instantiate the appropriate algorithm based on config,
    then delegates sampling to that algorithm's sample_from_uSet() method.
    Initializes algorithm-specific state (e.g., DCoM delta tracking) automatically.
    """

    def __init__(self, dataObj, cfg, lset=None, plotting_manager=None, annotator_manager=None):
        self.cfg = cfg
        self.annotator_manager = annotator_manager
        # Create the appropriate algorithm instance using factory
        self.al_algorithm = self._create_algorithm(lset, dataObj, plotting_manager)

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None,
                        checkpoint_file=None, num_samples=None, annotator_id=None):
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

        if isinstance(self.al_algorithm, CoverageMatrixMethod):
            activeSet, remain_set = self.al_algorithm.sample_from_uSet(
                clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
                checkpoint_file=checkpoint_file, num_samples=num_samples,
                annotator_id=annotator_id
            )
        else:
            activeSet, remain_set = self.al_algorithm.sample_from_uSet(
                clf_model, lSet, uSet, trainDataset, supportingModels=supportingModels,
                checkpoint_file=checkpoint_file, num_samples=num_samples
            )

        return activeSet, remain_set

    def _observed_train_labels(self):
        """Full-length observed-label array for samplers that consume labels
        in-loop from a table (BAYES_MISP). Uses the annotator's precomputed
        noisy_labels so label noise is respected; at noise rate 0 this equals
        the clean targets."""
        am = self.annotator_manager
        if am is None or not getattr(am, 'annotators', None):
            raise ValueError(
                "bayes_misp needs an observed-label table but no annotator is "
                "configured. Enable ANNOTATORS (single mode) — the BAYESCOV_* "
                "configs do this, including a rate-0.0 clean annotator.")
        if len(am.annotators) != 1:
            raise NotImplementedError(
                "bayes_misp port supports a single annotator only; got "
                f"{len(am.annotators)}. Its greedy loop reads one fixed label "
                "table, so per-point annotator choice cannot be honoured.")
        return np.asarray(am.annotators[0].noisy_labels)

    def _create_algorithm(self, lset=None, dataObj=None, plotting_manager=None):
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
            cmm = CoverageMatrixMethod(self.cfg, annotator_manager=self.annotator_manager)
            # TRAIN_LABEL_MODE != 'observed' (§4.8 probcover × pseudo control):
            # ProbCover has no posterior, so attach the same MispPosteriorEngine
            # used for bayes_misp (combine follows BAYES_COVERAGE.COMBINE,
            # 'product' by default = PoE training targets).
            train_label_mode = str(getattr(
                getattr(self.cfg, 'BAYES_COVERAGE', None), 'TRAIN_LABEL_MODE',
                'observed'))
            if train_label_mode != 'observed':
                import copy as _copy
                from .bayesian_coverage import MispPosteriorEngine
                eng_cfg = _copy.deepcopy(self.cfg)
                eng_cfg['SPARSE_K'] = False
                eng_cfg['K_SPARSE'] = False
                engine = MispPosteriorEngine(eng_cfg,
                                             annotator_manager=self.annotator_manager)
                return ProbCoverPseudoAdapter(cmm, engine)
            return cmm

        # NPC — Noise-aware ProbCover (Netta's thesis). Same coverage graph as
        # ProbCover (TopHat + sparse) wrapped in the Noise-Aware Sampling loop.
        elif sampling_fn in ["npc", "noise_aware_probcover", "noise_aware_prob_cover"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'tophat':
                logger.info('Warning!! for NPC the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                print('Warning!! for NPC the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')

                logger.info('Warning!! Running NPC as Sparse. Changing K_sparse to True')
                print('Warning!! Running NPC as Sparse. Changing K_sparse to True')

                self.cfg['KERNEL_TYPE'] = 'tophat'
                self.cfg['K_SPARSE'] = True
            if str(self.cfg.NOISE_FILTER.METHOD).lower() in ('none', 'noop', ''):
                raise ValueError(
                    "NPC requires a noise filter, but NOISE_FILTER.METHOD is 'none'. "
                    "Set NOISE_FILTER.METHOD=aum (or ideal); with 'none' NPC degenerates "
                    "to plain ProbCover.")
            noise_filter = build_noise_filter(self.cfg)
            return NoiseAwareCoverage(self.cfg, annotator_manager=self.annotator_manager,
                                      noise_filter=noise_filter)

        # Bayesian coverage (soft per-class posterior; design doc
        # docs/bayesian_coverage_DESIGN.md). Shares the coverage graph with
        # ProbCover (TopHat kernel) but replaces the binary max-coverage matrix
        # with a per-class posterior. bayes_product / bayes_additive are
        # convenience aliases that pin BAYES_COVERAGE.COMBINE for the design §3
        # product-vs-additive comparison; bayes_coverage honours the config value.
        elif sampling_fn in ["bayes_coverage", "bayes_product", "bayes_additive"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'tophat':
                logger.info('Warning!! for BayesianCoverage the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                print('Warning!! for BayesianCoverage the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                self.cfg['KERNEL_TYPE'] = 'tophat'
            # The sparse-K path is not implemented for the dense [M, num_classes]
            # posterior accumulator yet (design §7); force it off.
            self.cfg['SPARSE_K'] = False
            self.cfg['K_SPARSE'] = False
            if sampling_fn == "bayes_product":
                self.cfg.BAYES_COVERAGE.COMBINE = 'product'
            elif sampling_fn == "bayes_additive":
                self.cfg.BAYES_COVERAGE.COMBINE = 'additive'
            return BayesianCoverage(self.cfg, annotator_manager=self.annotator_manager)

        # Itai David's BAYES_MISP (github.com/itaidavid1/bayes_al_2025), vendored
        # port (BAYES_MISP.py + kernel_utils_misp.py + prior_selection.py). His
        # CIFAR100 benchmark config: tophat kernel, delta=0.75, diff_method='max',
        # sparse CSR K (the 'max' scoring path calls batched_diffs_sparse_max,
        # which hard-requires a CSR K — dense K crashes). The object is stateful
        # across episodes (C_general posterior persists), matching his pipeline
        # where the AL object is constructed once per run.
        elif sampling_fn in ["bayes_misp"]:
            if getattr(self.cfg, 'KERNEL_TYPE', None) != 'tophat':
                logger.info('Warning!! for BAYES_MISP (as benchmarked) the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                print('Warning!! for BAYES_MISP (as benchmarked) the Kernel must be TopHat. Changing KERNEL_TYPE to tophat')
                self.cfg['KERNEL_TYPE'] = 'tophat'
            if not self.cfg.get('SPARSE_K', False):
                logger.info("Warning!! BAYES_MISP diff_method='max' requires a CSR K matrix. Forcing SPARSE_K=True")
                print("Warning!! BAYES_MISP diff_method='max' requires a CSR K matrix. Forcing SPARSE_K=True")
                self.cfg['SPARSE_K'] = True
            _ensure_bayes_misp_cfg_defaults(self.cfg)
            from .BAYES_MISP import BAYES_MISP  # lazy: touches CUDA at import
            misp = BAYES_MISP(
                self.cfg,
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                train_labels=self._observed_train_labels(),
                lset=np.asarray(lset if lset is not None else [], dtype=int),
                delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA,
            )
            # TRAIN_LABEL_MODE != 'observed' (e.g. the §7.5 pseudo arm) needs the
            # BayesianCoverage posterior machinery, which BAYES_MISP doesn't have.
            # Attach a MispPosteriorEngine (dense-free) built on a cfg copy with
            # the sparse flags off — the engine path is dense-row. The engine's
            # combine follows BAYES_COVERAGE.COMBINE (CLI --combine): 'product'
            # = the §4.8 misp-hybrid arm, 'additive' = the misp_pure arm
            # (categorical training targets matching misp's own posterior family).
            engine = None
            train_label_mode = str(getattr(
                getattr(self.cfg, 'BAYES_COVERAGE', None), 'TRAIN_LABEL_MODE',
                'observed'))
            if train_label_mode != 'observed':
                import copy as _copy
                from .bayesian_coverage import MispPosteriorEngine
                eng_cfg = _copy.deepcopy(self.cfg)
                eng_cfg['SPARSE_K'] = False
                eng_cfg['K_SPARSE'] = False
                engine = MispPosteriorEngine(eng_cfg,
                                             annotator_manager=self.annotator_manager)
            return BayesMispAdapter(misp, posterior_engine=engine)

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
            cmm = CoverageMatrixMethod(self.cfg, annotator_manager=self.annotator_manager)
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


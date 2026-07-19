"""SupCon fine-tuning on top of an existing self-supervised encoder, with an
optional kernel-label-propagation variant, all measured on the same yardstick.

Question this answers: take an already-trained SimCLR (or BYOL) backbone, hand a
small labeled set to SupCon, and see whether the supervised contrastive pass
improves the representation -- and whether adding kernel-propagated pseudo-labels
improves it further. Three conditions are trained/evaluated identically:

    baseline      the base encoder as-is (no SupCon, no labels).
    supcon_only   base encoder -> SupCon on the labeled set only.
    supcon_prop   base encoder -> SupCon on labeled + confident pseudo-labels
                  (kernel label propagation over the base features).

Every condition is scored with the SAME offline linear-probe + weighted-kNN
(val_acc1/5, val_knn_acc1/5); the two trained conditions also stream solo's
native online val metrics to wandb during training.

Outputs go to a navigable tree, dataset first then embedding, then condition,
then (for propagation) kernel and hyperparameters:

    <results_root>/<dataset>/<embedding>/
        base_features/                          cached frozen features (shared)
        baseline/scores.json                    just the embedding
        supcon_only/L<L>_s<seed>/               embedding + supcon
        supcon_prop/<kernel>/bw<bw>_sim<st>_conf<ct>/L<L>_s<seed>/   + propagation

Baseline uses no labels, so it is identical across budgets: it is computed once
per (embedding, dataset) and the features are cached for the budget runs to
reuse. --mode controls which part runs:
    baseline   extract + cache features + score baseline, then stop.
    budget     load cached features, run supcon_only + supcon_prop for one budget.
    all        do both in one process (handy for a single local run).

Run one case end to end locally:

    python -m ssl_framework.algorithms.supcon_finetune \
        --dataset cifar100 \
        --base_checkpoint ../solo-learn/trained_models/simclr/cifar100/1000ep \
        --n_labeled 1000 --per_class_budget 10 --with_propagation --mode all

Nearly everything is a CLI knob; see main() for the full list and defaults.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..backends import solo_learn
from ..labeling import RandomSelector, class_counts
from .. import propagation as prop
from .. import evaluation as ev

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
_METRIC_KEYS = ["val_acc1", "val_acc5", "val_knn_acc1", "val_knn_acc5"]


# --------------------------------------------------------------------------- #
# small formatting / path helpers                                             #
# --------------------------------------------------------------------------- #
def _g(x):
    """Compact float for folder/run names: 0.0->'0', 0.1->'0.1', 9.2916->'9.29'."""
    return f"{float(x):g}"


def _floats(s):
    return [float(x) for x in str(s).split(",") if str(x) != ""]


def _bw_tokens(spec, median):
    """Parse a bandwidth spec (comma list) into [(token, value)]. Each token is
    'auto', a number (absolute h), or '<k>x' (k*median)."""
    out = []
    for t in [t.strip() for t in str(spec).split(",") if t.strip()]:
        if t == "auto":
            out.append(("auto", median))
        elif t.endswith("x"):
            out.append((t, float(t[:-1]) * median))
        else:
            out.append((t, float(t)))
    return out


def _budget_tag(args):
    return f"L{args.per_class_budget}" if args.per_class_budget is not None \
        else f"n{args.n_labeled}"


def _case_root(args, base_method):
    """<results_root>/<dataset>/<embedding> -- the shared root for this case."""
    return Path(args.results_root) / args.dataset / base_method


def _prop_hparam_dir(bw_tok, sim, conf):
    return f"bw{bw_tok}_sim{_g(sim)}_conf{_g(conf)}"


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# --------------------------------------------------------------------------- #
# wandb helpers (kept tiny so the rest reads cleanly)                          #
# --------------------------------------------------------------------------- #
def _make_logger(args, *, name, group, job_type, tags, save_dir):
    if not args.wandb:
        return None
    from lightning.pytorch.loggers import WandbLogger
    return WandbLogger(name=name, project=args.wandb_project, entity=args.wandb_entity,
                       group=group, job_type=job_type, tags=tags,
                       offline=args.wandb_offline, save_dir=str(save_dir),
                       reinit=True)


def _log(logger, metrics):
    if logger is not None:
        logger.log_metrics({k: v for k, v in metrics.items()})


def _log_config(logger, cfg):
    """Put run identity (base encoder, dataset, budget, kernel, thresholds) into
    the wandb run config, and register per_class_budget as a custom x-axis so it
    can be selected in line panels (wandb otherwise only offers step/epoch)."""
    if logger is not None:
        try:
            exp = logger.experiment
            exp.config.update({k: v for k, v in cfg.items() if v is not None},
                              allow_val_change=True)
            exp.define_metric("per_class_budget")
            for pat in ("offline/*", "baseline/*", "improvement/*",
                        "n_pseudo", "n_train", "pseudo_accuracy", "pseudo_fraction"):
                exp.define_metric(pat, step_metric="per_class_budget")
        except Exception:
            pass


def _finish(logger):
    if logger is not None:
        try:
            logger.experiment.finish()
        except Exception:
            pass


def _log_makeup(logger, counts):
    if logger is None:
        return
    try:
        import wandb
        table = wandb.Table(data=[[int(c), int(n)] for c, n in enumerate(counts)],
                            columns=["class", "count"])
        logger.experiment.log({"labeled/class_counts": wandb.plot.bar(table, "class", "count")})
    except Exception:
        pass
    _log(logger, {"labeled/n_empty_classes": int((counts == 0).sum()),
                  "labeled/min_per_class": int(counts.min()),
                  "labeled/max_per_class": int(counts.max()),
                  "labeled/std_per_class": float(counts.std())})


# --------------------------------------------------------------------------- #
# training / evaluation of one condition                                      #
# --------------------------------------------------------------------------- #
def _recommend_epochs(n_train, batch_size, target_steps, override, max_epochs):
    """Fine-tuning is measured in gradient steps, not epochs: an "epoch" over a
    small labeled set is only ceil(n/batch) steps. Pick epochs to hit ~target
    steps, but CAP the epoch count -- otherwise a tiny labeled set (1 step/epoch)
    would run thousands of one-step epochs (slow, and over-trains) to reach the
    step target. The cap trades the step target for a sane number of passes."""
    if override is not None:
        return int(override)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    return max(1, min(max_epochs, round(target_steps / steps_per_epoch)))


def _run_supcon_condition(name, *, dataset, base_state, train_indices, label_overrides,
                          data_dir, num_workers, args, logger):
    n_train = len(train_indices)
    batch_size = min(args.supcon_batch_size, n_train)
    epochs = _recommend_epochs(n_train, batch_size, args.supcon_target_steps,
                               args.supcon_epochs, args.supcon_max_epochs)
    warmup = args.supcon_warmup_epochs if args.supcon_warmup_epochs is not None \
        else min(10, max(1, epochs // 10))
    val_every = args.val_every or max(1, epochs // 20)
    print(f"[{name}] SupCon on {n_train} images | batch={batch_size} "
          f"epochs={epochs} (~{epochs * math.ceil(n_train / batch_size)} steps) "
          f"warmup={warmup} val_every={val_every}")

    result = solo_learn.run_phase(
        "supcon", dataset, epochs,
        data_dir=data_dir, init_backbone_state=base_state,
        labeled_indices=train_indices, label_overrides=label_overrides,
        batch_size=batch_size, num_workers=num_workers,
        enable_eval=True, knn_k=args.knn_k, knn_distance=args.knn_distance,
        val_every=val_every, lr=args.supcon_lr, classifier_lr=args.supcon_classifier_lr,
        warmup_epochs=warmup, logger=logger)
    return result, {"epochs": epochs, "batch_size": batch_size, "n_train": n_train}


def _evaluate_backbone(backbone, dataset, data_dir, num_workers, args):
    """Extract train+val features and score them with the shared yardstick."""
    import torch
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    emb = solo_learn.extract_embeddings(backbone, dataset, data_dir, num_workers=num_workers)
    (xtr, ytr), (xva, yva) = emb["train"], emb["val"]
    scores = ev.evaluate_features(xtr, ytr, xva, yva, knn_k=args.knn_k,
                                  knn_distance=args.knn_distance, probe_epochs=args.probe_epochs)
    return scores, emb


def _score_features(emb, args):
    (xtr, ytr), (xva, yva) = emb["train"], emb["val"]
    return ev.evaluate_features(xtr, ytr, xva, yva, knn_k=args.knn_k,
                                knn_distance=args.knn_distance, probe_epochs=args.probe_epochs)


# --------------------------------------------------------------------------- #
# baseline: compute once per (embedding, dataset), cache the features          #
# --------------------------------------------------------------------------- #
def _load_cached_features(features_dir):
    f = features_dir / "train_features.npy"
    if not f.exists():
        return None
    return {
        "train": (np.load(features_dir / "train_features.npy"),
                  np.load(features_dir / "train_labels.npy")),
        "val": (np.load(features_dir / "val_features.npy"),
                np.load(features_dir / "val_labels.npy")),
    }


def _metrics_with_reference(scores, base, budget, n_labeled, extra=None):
    """Metrics for a supcon run: its own offline scores, plus -- so a single plot
    shows the improvement over the embedding alone -- the baseline value as a flat
    reference (baseline/<m>) and the gain (improvement/<m> = supcon - baseline)."""
    out = {**{f"offline/{k}": v for k, v in scores.items()},
           "per_class_budget": budget, "n_labeled": n_labeled}
    if base:
        out.update({f"baseline/{k}": base[k] for k in scores})
        out.update({f"improvement/{k}": scores[k] - base[k] for k in scores})
    if extra:
        out.update(extra)
    return out


def _load_baseline_scores(case):
    """The embedding-alone offline scores written by the baseline job, or None."""
    f = case / "baseline" / "scores.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("offline")
        except Exception:
            return None
    return None


def _compute_baseline(args, base_backbone, base_method, case, features_dir):
    """Extract + cache the base features and score the raw encoder (no wandb).

    Logging is done separately (_log_baseline) so the baseline run can be
    re-logged on later launches from the cache without re-extracting.
    """
    scores, emb = _evaluate_backbone(base_backbone, args.dataset, args.data_dir,
                                     args.num_workers, args)
    features_dir.mkdir(parents=True, exist_ok=True)
    for split, (f, y) in emb.items():
        np.save(features_dir / f"{split}_features.npy", f)
        np.save(features_dir / f"{split}_labels.npy", y)
    _write_json(case / "baseline" / "scores.json",
                {"base_method": base_method, "dataset": args.dataset,
                 "condition": "baseline", "offline": scores})
    print(f"[baseline] {scores}")
    return emb, scores


def _log_baseline(args, base_method, case, scores):
    """Log the baseline wandb run. The baseline uses no labels, so its accuracy is
    one number; to make it a flat REFERENCE LINE across the budget axis (so the
    improvement is visible as the gap above it), we log that value at every budget
    in --reference_budgets. Called on EVERY baseline launch -- including when the
    features are cached -- so the reference line always spans the full budget range.
    """
    run_cfg = {"base_method": base_method, "dataset": args.dataset, "condition": "baseline"}
    logger = _make_logger(args, name=f"{base_method}-{args.dataset}-baseline",
                          group=f"{base_method}-{args.dataset}", job_type="baseline",
                          tags=[base_method, args.dataset, "baseline"],
                          save_dir=case / "baseline")
    _log_config(logger, run_cfg)
    ref_budgets = [int(x) for x in str(args.reference_budgets).split(",") if x.strip()]
    offline = {f"offline/{k}": v for k, v in scores.items()}
    if ref_budgets:
        for L in ref_budgets:  # flat line: same value at each budget x
            _log(logger, {**offline, "per_class_budget": L})
    else:
        _log(logger, offline)
    _finish(logger)


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def run(args):
    solo_learn.set_seed(args.seed)
    num_classes = _NUM_CLASSES[args.dataset]
    data_dir = args.data_dir
    want_baseline = args.mode in ("baseline", "all")
    want_budget = args.mode in ("budget", "all")

    # ---- base encoder + case root ----------------------------------------- #
    base_method = solo_learn.checkpoint_method(args.base_checkpoint)  # simclr | byol
    case = _case_root(args, base_method)
    features_dir = case / "base_features"
    print(f"case={case} mode={args.mode} base={base_method}")

    base_backbone = solo_learn.load_backbone_from_checkpoint(args.base_checkpoint)
    base_state = {k: v.detach().cpu().clone() for k, v in base_backbone.state_dict().items()}

    # ---- baseline: extract+cache once, but LOG the reference run every launch  #
    # (even when cached) so the baseline flat line always spans the full budget
    # range -- otherwise a relaunch that reuses the cache would leave the baseline
    # showing only whatever budgets the first launch happened to log.
    base_emb = _load_cached_features(features_dir)
    if want_baseline:
        if base_emb is None or args.force_baseline:
            base_emb, base_scores = _compute_baseline(args, base_backbone, base_method,
                                                      case, features_dir)
        else:
            base_scores = _load_baseline_scores(case)
            print(f"[baseline] cached {base_scores}")
        if base_scores is not None:
            _log_baseline(args, base_method, case, base_scores)
    if args.mode == "baseline":
        print("done (baseline) ->", case)
        return

    if base_emb is None:
        raise SystemExit(
            f"no cached base features at {features_dir}; run --mode baseline first "
            "(or --mode all) for this embedding+dataset before budget runs")

    if not want_budget:
        return

    # embedding-alone scores, for the reference line + improvement on every run.
    baseline_scores = _load_baseline_scores(case)

    # ---- labeled set + its class makeup ----------------------------------- #
    n_pool = solo_learn.train_pool_size(args.dataset, data_dir)
    targets = solo_learn.train_targets(args.dataset, data_dir)
    labeled = RandomSelector().select(n_pool, args.n_labeled, args.seed)
    counts = class_counts(labeled, targets, num_classes)
    makeup = {
        "counts": counts.tolist(), "n_labeled": int(len(labeled)),
        "n_empty_classes": int((counts == 0).sum()), "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()), "std_per_class": float(counts.std()),
    }
    # The per-class makeup of the (random, unbalanced) labeled set -- saved to a
    # dedicated file per budget, logged to wandb as a bar chart + summary stats,
    # and also embedded in supcon_only/scores.json.
    _write_json(case / "labeled" / f"{_budget_tag(args)}_s{args.seed}_class_counts.json",
                {"base_method": base_method, "dataset": args.dataset,
                 "per_class_budget": args.per_class_budget, **makeup})
    print(f"pool={n_pool} labeled={len(labeled)} {_budget_tag(args)} | classes: "
          f"empty={makeup['n_empty_classes']} min={makeup['min_per_class']} "
          f"max={makeup['max_per_class']} std={makeup['std_per_class']:.2f} "
          f"-> {case / 'labeled'}")

    budget = args.per_class_budget
    group = f"{base_method}-{args.dataset}"
    id_cfg = {"base_method": base_method, "dataset": args.dataset,
              "per_class_budget": budget, "n_labeled": args.n_labeled, "seed": args.seed}

    # ---- condition: supcon_only ------------------------------------------- #
    leaf = case / "supcon_only" / f"{_budget_tag(args)}_s{args.seed}"
    logger = _make_logger(args, name=f"{group}-supcon_only-{_budget_tag(args)}-s{args.seed}",
                          group=group, job_type="supcon_only",
                          tags=[base_method, args.dataset, "supcon_only", _budget_tag(args)],
                          save_dir=leaf)
    _log_config(logger, {**id_cfg, "condition": "supcon_only"})
    _log_makeup(logger, counts)
    res, meta = _run_supcon_condition("supcon_only", dataset=args.dataset, base_state=base_state,
                                      train_indices=labeled, label_overrides=None,
                                      data_dir=data_dir, num_workers=args.num_workers,
                                      args=args, logger=logger)
    scores, _ = _evaluate_backbone(res["model"].backbone, args.dataset, data_dir,
                                   args.num_workers, args)
    print(f"[supcon_only] {scores}")
    _log(logger, _metrics_with_reference(scores, baseline_scores, budget, args.n_labeled))
    _finish(logger)
    leaf.mkdir(parents=True, exist_ok=True)
    np.save(leaf / "labeled_indices.npy", np.array(labeled))
    solo_learn.save_backbone_state(res["backbone_state"], leaf / "backbone.pt")
    _write_json(leaf / "scores.json", {**id_cfg, "condition": "supcon_only",
                                       "offline": scores, "class_makeup": makeup, **meta})

    # ---- condition(s): supcon_prop ---------------------------------------- #
    if args.with_propagation:
        xtr = base_emb["train"][0]  # base encoder features drive the propagation
        kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
        median = prop.median_bandwidth(xtr)

        if args.auto_thresholds > 0:
            # Figure out good thresholds automatically: rank the whole grid on
            # the cached features (cheap, no training), then train the top-K per
            # kernel that add positive, pure net supervision. This is how the
            # thresholds are chosen to be "good" -- and it trains several, so you
            # see the comparison rather than trusting one hand-picked setting.
            from . import threshold_search as ts
            gbws = _bw_tokens(args.auto_grid_bandwidths, median)
            rows = ts.rank_combos(xtr, targets, labeled, num_classes, kernels=kernels,
                                  bandwidths=gbws, sims=_floats(args.auto_grid_sims),
                                  confs=_floats(args.auto_grid_confs), agg=args.agg,
                                  confidence_metric=args.confidence_metric,
                                  wrong_penalty=args.auto_wrong_penalty)
            _write_json(case / "threshold_search" / f"{_budget_tag(args)}_s{args.seed}.json",
                        {**id_cfg, "median_distance": median, "grid": rows})
            chosen = ts.select_per_kernel(rows, kernels, per_kernel=args.auto_thresholds,
                                          min_purity=args.auto_min_purity)
            print(f"auto-thresholds: {len(chosen)} combos "
                  f"(top-{args.auto_thresholds}/kernel, purity>={args.auto_min_purity}, net>0); "
                  f"median dist={median:.4g}")
            for r in chosen:
                bw_tok = "na" if r["kernel"] == "cosine" else r["bw_token"]
                _, bandwidth = _bw_tokens(bw_tok if bw_tok != "na" else "auto", median)[0]
                _one_prop(args, case, group, base_method, budget, id_cfg,
                          xtr, targets, labeled, num_classes, base_state, data_dir,
                          r["kernel"], bw_tok, bandwidth, r["sim_threshold"], r["conf_threshold"],
                          baseline_scores)
            trained = {r["kernel"] for r in chosen}
            for k in kernels:
                if k not in trained:
                    print(f"  ({k}: no combo cleared purity>={args.auto_min_purity} with "
                          "net>0 at this budget -- skipped)")
        else:
            # Explicit list-valued thresholds: every (kernel, bw, sim, conf) is
            # trained as its own supcon_prop leaf + wandb run.
            bws = _bw_tokens(args.bandwidth, median)
            sims = _floats(args.sim_threshold)
            confs = _floats(args.conf_threshold)
            print(f"propagation kernels={kernels} bandwidths={[t for t,_ in bws]} "
                  f"sims={sims} confs={confs} (median dist={median:.4g})")
            for kernel in kernels:
                kbws = [bws[0]] if kernel == "cosine" else bws  # cosine ignores bandwidth
                for bw_tok, bandwidth in kbws:
                    for sim in sims:
                        for conf in confs:
                            _one_prop(args, case, group, base_method, budget, id_cfg,
                                      xtr, targets, labeled, num_classes, base_state, data_dir,
                                      kernel, bw_tok, bandwidth, sim, conf, baseline_scores)

    print("done (budget) ->", case)


def _one_prop(args, case, group, base_method, budget, id_cfg, xtr, targets, labeled,
              num_classes, base_state, data_dir, kernel, bw_tok, bandwidth, sim, conf,
              baseline_scores=None):
    """Propagate + (if any pseudo-labels) train one supcon_prop combination."""
    pr = prop.propagate(xtr, labeled, targets[labeled], num_classes,
                        kernel=kernel, bandwidth=bandwidth, sim_threshold=sim,
                        conf_threshold=conf, agg=args.agg,
                        confidence_metric=args.confidence_metric,
                        max_pseudo=args.max_pseudo, true_targets=targets)
    tag = f"{kernel} bw={bw_tok} sim={_g(sim)} conf={_g(conf)}"
    print(f"[supcon_prop/{tag}] propagation {pr.stats}")
    pleaf = (case / "supcon_prop" / kernel / _prop_hparam_dir(bw_tok, sim, conf)
             / f"{_budget_tag(args)}_s{args.seed}")
    pleaf.mkdir(parents=True, exist_ok=True)
    np.save(pleaf / "pseudo_indices.npy", pr.pseudo_indices)
    np.save(pleaf / "pseudo_labels.npy", pr.pseudo_labels)
    _write_json(pleaf / "propagation_stats.json", pr.stats)

    pcfg = {**id_cfg, "condition": "supcon_prop", "kernel": kernel,
            "bandwidth": bandwidth, "bandwidth_token": bw_tok, "sim_threshold": sim,
            "conf_threshold": conf, "agg": args.agg, "confidence_metric": args.confidence_metric}

    if pr.pseudo_indices.size == 0:
        print(f"[supcon_prop/{tag}] no pseudo-labels cleared the threshold; "
              "skipping (identical to supcon_only)")
        _write_json(pleaf / "scores.json", {**pcfg, "offline": None,
                                            "propagation": pr.stats, "skipped": "no_pseudo_labels"})
        return

    train_idx = sorted(set(labeled) | set(int(i) for i in pr.pseudo_indices))
    overrides = {int(i): int(l) for i, l in zip(pr.pseudo_indices, pr.pseudo_labels)}

    rname = (f"{group}-supcon_prop-{kernel}-{_budget_tag(args)}-"
             f"bw{bw_tok}-sim{_g(sim)}-conf{_g(conf)}-s{args.seed}")
    logger = _make_logger(args, name=rname, group=group, job_type="supcon_prop",
                          tags=[base_method, args.dataset, "supcon_prop", kernel,
                                _budget_tag(args)],
                          save_dir=pleaf)
    _log_config(logger, pcfg)
    _log(logger, {f"propagation/{k}": v for k, v in pr.stats.items()
                  if isinstance(v, (int, float))})
    res, meta = _run_supcon_condition(f"supcon_prop/{kernel}", dataset=args.dataset,
                                      base_state=base_state, train_indices=train_idx,
                                      label_overrides=overrides, data_dir=data_dir,
                                      num_workers=args.num_workers, args=args, logger=logger)
    scores, _ = _evaluate_backbone(res["model"].backbone, args.dataset, data_dir,
                                   args.num_workers, args)
    print(f"[supcon_prop/{tag}] {scores}")
    _log(logger, _metrics_with_reference(scores, baseline_scores, budget, args.n_labeled,
                 extra={"pseudo_accuracy": pr.stats.get("pseudo_accuracy", float("nan")),
                        "n_pseudo": pr.stats["n_pseudo"],          # points ADDED by propagation
                        "n_train": meta["n_train"],                # labeled + pseudo
                        "pseudo_fraction": pr.stats["n_pseudo"] / max(1, meta["n_train"])}))
    _finish(logger)
    solo_learn.save_backbone_state(res["backbone_state"], pleaf / "backbone.pt")
    _write_json(pleaf / "scores.json", {**pcfg, "offline": scores, "propagation": pr.stats, **meta})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / base encoder / layout
    p.add_argument("--dataset", default="cifar100", choices=list(_NUM_CLASSES))
    p.add_argument("--base_checkpoint", required=True,
                   help="run dir (args.json + *.ckpt) or a .ckpt of the base encoder")
    p.add_argument("--results_root", default=str(Path(__file__).resolve().parents[1] / "results"),
                   help="root of the results tree (default ssl_framework/results)")
    p.add_argument("--mode", default="all", choices=["baseline", "budget", "all"],
                   help="baseline: extract+cache+score encoder; budget: supcon runs "
                        "for one budget (needs cached features); all: both")
    p.add_argument("--force_baseline", action="store_true",
                   help="recompute+recache base features even if present")
    p.add_argument("--reference_budgets", default="",
                   help="comma list of budgets L; the baseline logs its (label-free) "
                        "score at each so it draws a flat reference line across the "
                        "budget axis in wandb")
    p.add_argument("--data_dir", default=str(solo_learn.DEFAULT_DATA_DIR))
    p.add_argument("--num_workers", type=int, default=4)
    # labeled set
    p.add_argument("--n_labeled", type=int, default=0,
                   help="total labeled-set size (random/unbalanced); required for budget runs")
    p.add_argument("--per_class_budget", type=int, default=None,
                   help="nominal labels-per-class L (= n_labeled / num_classes); the "
                        "shared x-axis for cross-run plots (metadata; keep consistent with n_labeled)")
    p.add_argument("--seed", type=int, default=1)
    # supcon fine-tuning (measured in steps; epochs auto-derived unless pinned)
    p.add_argument("--supcon_epochs", type=int, default=None,
                   help="pin epochs; default auto from --supcon_target_steps")
    p.add_argument("--supcon_target_steps", type=int, default=5000,
                   help="gradient-step target; epochs = target/steps_per_epoch, capped by "
                        "--supcon_max_epochs so tiny budgets don't run thousands of 1-step epochs")
    p.add_argument("--supcon_max_epochs", type=int, default=200,
                   help="hard cap on epochs (passes over the labeled set); keeps tiny "
                        "budgets fast and avoids over-training")
    p.add_argument("--supcon_batch_size", type=int, default=256)
    p.add_argument("--supcon_lr", type=float, default=0.1,
                   help="fine-tune LR (< the from-scratch 0.4); scaled by batch/256")
    p.add_argument("--supcon_classifier_lr", type=float, default=0.1)
    p.add_argument("--supcon_warmup_epochs", type=int, default=None)
    p.add_argument("--val_every", type=int, default=0,
                   help="online validation every N epochs (0 = auto, ~20 points)")
    # evaluation
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--knn_distance", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--probe_epochs", type=int, default=100)
    # propagation
    p.add_argument("--with_propagation", action="store_true")
    p.add_argument("--kernels", default="gaussian",
                   help="comma-separated kernels as separate supcon_prop conditions "
                        f"(any of {','.join(prop.KERNELS)}); supcon_only is shared")
    p.add_argument("--bandwidth", default="auto",
                   help="comma list; each 'auto' (median), a number (absolute h), or "
                        "'<k>x' (k*median). gaussian/laplacian only; ignored for cosine")
    p.add_argument("--sim_threshold", default="0.0",
                   help="comma list; kernel floor(s): pairs below add 0 to the coverage matrix")
    p.add_argument("--conf_threshold", default="0.9",
                   help="comma list; min confidence(s) for a point to be pseudo-labeled")
    p.add_argument("--agg", default="max", choices=list(prop.AGGREGATIONS))
    p.add_argument("--confidence_metric", default="prob", choices=list(prop.CONFIDENCE_METRICS))
    p.add_argument("--max_pseudo", type=int, default=None)
    # automatic threshold selection (overrides the explicit lists above)
    p.add_argument("--auto_thresholds", type=int, default=0,
                   help="if >0, rank the threshold grid on the cached features and "
                        "train the top-N per kernel (positive net-correct, purity floor) "
                        "instead of the explicit --bandwidth/--sim/--conf lists")
    p.add_argument("--auto_grid_bandwidths", default="0.5x,0.25x,0.1x",
                   help="tighter bandwidths give purer pseudo-labels; add 'auto' for the "
                        "wide (median) end")
    p.add_argument("--auto_grid_sims", default="0.1,0.2,0.3",
                   help="higher figures floors cut weak, impure votes")
    p.add_argument("--auto_grid_confs", default="0.8,0.9,0.95")
    p.add_argument("--auto_min_purity", type=float, default=0.9,
                   help="reject auto-chosen combos whose pseudo-label purity is below this; "
                        "raise toward 0.95 for cleaner (fewer) pseudo-labels")
    p.add_argument("--auto_wrong_penalty", type=float, default=3.0,
                   help="lambda in net_correct = n*(purity - lambda*(1-purity)); higher = "
                        "the ranking punishes wrong labels harder, favouring purity over reach")
    # wandb
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", default="semi-supervised-supcon")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    args = p.parse_args()
    if args.mode in ("budget", "all") and args.n_labeled <= 0:
        p.error("--n_labeled must be > 0 for --mode budget/all")
    run(args)


if __name__ == "__main__":
    main()

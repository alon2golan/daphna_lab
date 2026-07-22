#!/usr/bin/env python3
"""Build the single source-of-truth results CSV for the bayesian-coverage study.

One row per *fully-qualified per-seed cell*:

    sampling, model, train_labels, norm, noise, budget, seed, test_acc, sweep_id, source

- `sampling`      how points were selected              (random / probcover / bayes_additive / bayes_product / npc)
- `model`         how a test point is classified         (linear head / 1-NN / posterior-argmax / NW ...)
- `train_labels`  targets the classifier trained on      (observed / soft_posterior / ... / pseudo (all) / AUM clean-train / —)
- `norm`          1 iff L2-normalized classifier inputs
- `test_acc`      percent

This is the RAW truth layer (includes superseded runs, e.g. the buggy pre-fix
bayes_additive in sweep 30978366 — the renderer's ROW_SPEC pins additive to the
fixed rerun 31066809 and simply doesn't select the old rows). The MD/HTML tables
are a *pivot* of this file — see render_results_tables.py. Nothing downstream reads
result.json directly anymore.

Sources ingested:
  1. output/CIFAR100/**/result.json  for the 8 study sweeps  (trained arms + probcover 1-NN + Netta)
  2. results/cifar100/eval_1nn_all_arms/per_episode.csv       (training-free evaluators: 1-NN / NW; future kNN / soft-kNN land here)
  3. results/cifar100/posterior_eval/test_posterior_acc.json  (graph posterior alone = posterior-argmax, no training)

Usage:  python3 batch_utils/build_results_master.py
Writes: results/cifar100/results_master.csv
"""
import csv
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cifar100", "results_master.csv")

# The 8 sweeps that feed the §4 tables (Table R1).
STUDY_SWEEPS = {
    "30978366",  # Bayes comparison: {random, probcover, bayes_additive, bayes_product} selection baselines
    "31066809",  # bayes_additive fixed-acquisition rerun (supersedes additive rows in 30978366)
    "30980646",  # bayes_product train-modes: observed/soft/map/filtered/reweighted
    "31047129",  # bayes_product pseudo-labeling: all / lin10 / agg50
    "31048960",  # normalized-feature rerun (norm=1)
    "31048649",  # probcover 1-NN evaluator
    "30811243",  # Netta NPC repro
    "30863746",  # AUM clean-train ablation (Netta rows)
    "31078514",  # BAYES_MISP port parity check: probcover / bayes_misp / bayes_additive under the MLP head (clean, seed 1)
    "31084429",  # bayes_misp noise sweep: noise {0,.2,.5,.8} x seeds {1,2,3} x heads {MLP, linear}
    "31085570",  # bayes_misp MLP best-val rerun v1 — INVALID for display: partial_fit accidentally warm-started across episodes
    "31085802",  # bayes_misp MLP best-val rerun v2 (cold clone per episode) — the ROW_SPEC pin
    "31085867",  # classifier comparison: random + probcover under the MLP head (cold + best-val)
    "31085854",  # classifier comparison: pseudo-all x {LH-norm x rkl, torch-MLP x {softce, rkl}} + torch-MLP parity gate. LH rows VALID; torch-MLP rows INVALID for display (Adam wd 200x too strong — sklearn alpha semantics; parity gate failed 26.5->10.6)
    "31088628",  # wd-fix rerun of the torch-MLP arms (WEIGHT_DECAY 3e-2 -> 1.5e-4) — the display pin for MLP-torch rows
    "31086759",  # rerun of 31085867 tasks 5+7 (hung at training start) — normalized below
}

# A rerun of individual hung/failed array tasks gets its own SLURM job id, but the
# rows belong to the original sweep: normalize so downstream sweep pins stay single-id.
RERUN_OF = {
    "31086759": "31085867",
}

FIELDS = ["sampling", "model", "train_labels", "norm", "noise", "budget", "seed", "test_acc", "sweep_id", "source"]

# canonical pseudo_k schedules -> arm name
PSEUDO_K_LIN10 = "1,2,3,4,5,6,7,8,9,10"
PSEUDO_K_AGG50 = "1,2,3,5,8,13,20,30,40,50"


def model_of(p):
    return {"from_features": "linear head", "NN1": "1-NN",
            "from_mlp_sklearn": "MLP head",
            "from_mlp_torch": "MLP head (torch)"}.get(p.get("eval_model_type"), p.get("eval_model_type"))


def train_labels_of(p):
    """Map a run's params to a canonical Train-labels value (see the doc's §4.1 key).
    A non-default soft loss (--soft_loss rkl, §9.8) is suffixed onto the label so
    the two loss arms never collapse into one seed-group."""
    labels = _train_labels_of(p)
    if p.get("soft_loss") and p.get("soft_loss") != "softce":
        return f"{labels}, {p['soft_loss']}"
    return labels


def _train_labels_of(p):
    if p.get("eval_model_type") == "NN1":
        return "observed"  # the 1-NN evaluator is fitted on observed labels
    tlm = p.get("train_label_mode")
    if tlm == "pseudo":
        scope, pk = p.get("pseudo_scope"), p.get("pseudo_k")
        if scope == "all":
            return "pseudo (all)"
        if scope == "topk" and pk == PSEUDO_K_LIN10:
            return "pseudo (lin10)"
        if scope == "topk" and pk == PSEUDO_K_AGG50:
            return "pseudo (agg50)"
        return f"pseudo ({scope}:{pk})"
    if tlm in ("observed", "soft_posterior", "map_posterior", "filtered", "reweighted"):
        return tlm
    # tlm is None -> either a plain selection baseline (train on observed) or a Netta AUM arm
    if p.get("noise_filter") == "aum":
        return "AUM clean-train" if p.get("train_on_clean") else "observed (sel. only)"
    return "observed"


def rows_from_result_json():
    out = []
    for f in glob.glob(os.path.join(REPO, "output", "CIFAR100", "**", "result.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sid = d.get("sweep_id")
        if sid not in STUDY_SWEEPS:
            continue
        sid = RERUN_OF.get(sid, sid)
        p = d.get("params", {})
        base = dict(
            sampling=p.get("al_method"),
            model=model_of(p),
            train_labels=train_labels_of(p),
            norm=1 if p.get("normalize_features") else 0,
            noise=float(p.get("noise_rate") or 0.0),
            seed=p.get("seed"),
            sweep_id=sid,
            source="result.json",
        )
        for ep in d.get("episodes", []):
            b, acc = ep.get("labeled_size"), ep.get("test_acc")
            if b is None or acc is None:
                continue
            out.append({**base, "budget": int(b), "test_acc": round(float(acc), 3)})
    return out


def rows_from_eval_1nn():
    """Training-free evaluators. evaluator string -> (model, train_labels)."""
    path = os.path.join(REPO, "results", "cifar100", "eval_1nn_all_arms", "per_episode.csv")
    if not os.path.exists(path):
        return []
    EVAL_MAP = {
        "1nn_observed": ("1-NN", "observed"),
        "1nn_postargmax": ("1-NN", "posterior-argmax"),
        "nw_k1": ("NW (k=1)", "observed"),
        "nw_k3": ("NW (k=3)", "observed"),
    }
    out = []
    for r in csv.DictReader(open(path)):
        # bayes_additive here was replayed from the PRE-FIX acquisition (30978366,
        # clumped low-budget selection) → 1-NN reads absurdly low. It is recomputed
        # from the fixed sweep 31066809 in per_episode_fill.csv; skip the stale one.
        if r["arm"] == "bayes_additive":
            continue
        model, labels = EVAL_MAP.get(r["evaluator"], (r["evaluator"], "observed"))
        out.append(dict(
            sampling=r["arm"], model=model, train_labels=labels, norm=1,
            noise=float(r["noise"]), budget=int(r["budget"]), seed=int(r["seed"]),
            test_acc=round(float(r["test_acc"]), 3),
            sweep_id="eval_1nn_all_arms", source="per_episode.csv",
        ))
    return out


def rows_from_eval_1nn_fill():
    """1-NN fill/correction (eval_1nn_fill.py): fixed-sweep bayes_additive +
    npc (two AUM lineages). arm -> (sampling, train_labels); sweep_id from the
    file's src_sweep so provenance points at the real source sweep."""
    path = os.path.join(REPO, "results", "cifar100", "eval_1nn_all_arms", "per_episode_fill.csv")
    if not os.path.exists(path):
        return []
    ARM_MAP = {  # arm -> (sampling, train_labels)
        "bayes_additive": ("bayes_additive", "observed"),
        "npc_selonly": ("npc", "observed (sel. only)"),
        "npc_aum": ("npc", "AUM clean-train"),
    }
    out = []
    for r in csv.DictReader(open(path)):
        mapped = ARM_MAP.get(r["arm"])
        if mapped is None:
            continue
        sampling, labels = mapped
        out.append(dict(
            sampling=sampling, model="1-NN", train_labels=labels, norm=1,
            noise=float(r["noise"]), budget=int(r["budget"]), seed=int(r["seed"]),
            test_acc=round(float(r["test_acc"]), 3),
            sweep_id=r["src_sweep"], source="per_episode_fill.csv",
        ))
    return out


def rows_from_eval_heads_fill():
    """Classifier-comparison 1-NN fills (eval_heads_fill.py, gate-checked):
    bayes_misp 1-NN on observed labels (lSets of the 31084429 linear runs) +
    the pseudo-all TRAINING-SET 1-NN (lSet posterior-argmax + covered pool
    posterior-argmax — §8.3 Q2 variant (c))."""
    path = os.path.join(REPO, "results", "cifar100", "eval_heads_fill", "per_episode.csv")
    if not os.path.exists(path):
        return []
    ARM_MAP = {  # arm -> (sampling, model, train_labels)
        "bayes_misp": ("bayes_misp", "1-NN", "observed"),
        "bayes_product_pseudoall": ("bayes_product", "1-NN", "pseudo (all) targets"),
    }
    out = []
    for r in csv.DictReader(open(path)):
        mapped = ARM_MAP.get(r["arm"])
        if mapped is None:
            continue
        sampling, model, labels = mapped
        out.append(dict(
            sampling=sampling, model=model, train_labels=labels, norm=1,
            noise=float(r["noise"]), budget=int(r["budget"]), seed=int(r["seed"]),
            test_acc=round(float(r["test_acc"]), 3),
            sweep_id=r["src_sweep"], source="eval_heads_fill.csv",
        ))
    return out


def rows_from_posterior_alone():
    """Graph posterior alone: acc_all (fraction) -> percent; no trained model."""
    path = os.path.join(REPO, "results", "cifar100", "posterior_eval", "test_posterior_acc.json")
    if not os.path.exists(path):
        return []
    out = []
    for rec in json.load(open(path)):
        for b, cell in rec.get("budgets", {}).items():
            if "acc_all" not in cell:
                continue
            out.append(dict(
                sampling="bayes_product", model="posterior-argmax", train_labels="—", norm=1,
                noise=float(rec["noise"]), budget=int(b), seed=int(rec["seed"]),
                test_acc=round(float(cell["acc_all"]) * 100, 3),
                sweep_id="test_posterior_acc", source="test_posterior_acc.json",
            ))
    return out


def main():
    rows = (rows_from_result_json() + rows_from_eval_1nn()
            + rows_from_eval_1nn_fill() + rows_from_eval_heads_fill()
            + rows_from_posterior_alone())
    rows.sort(key=lambda r: (str(r["sampling"]), str(r["model"]), str(r["train_labels"]),
                             r["norm"], r["noise"], r["budget"], str(r["seed"])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {os.path.relpath(OUT, REPO)}")
    # brief provenance summary
    import collections
    by_src = collections.Counter(r["source"] for r in rows)
    print("  by source:", dict(by_src))


if __name__ == "__main__":
    main()

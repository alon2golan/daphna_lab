"""Run the pinned Netta NPC experiment on the lab's full frozen CIFAR-100 BYOL bank."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np


UPSTREAM_COMMIT = "733651bb0943a96392895103eb2ffc4e99a7c7d1"
UPSTREAM_SOURCE_SHA256 = "332f671b58881695340c13904c09ed48bc25a44600d63cc44ba7ff5823cf5ff5"
CONFIG = Path("configs/cifar100/al/RESNET18.yaml")
BANK_HASHES = {
    "train_features": "4e91020dcd15d77dcbe35ad32e63020bd156ad610808866ae49eb696d2035476",
    "train_targets": "233cdd4afc94cd00f3ddec6f8640168bf76a6872168cf6c5a9990ac9f144be3b",
    "test_features": "d7ba6cfb072039698da1b4e67fe97636ab0d4e6b6158bd9d66b42959bab18e36",
    "test_targets": "7827272d0fbeee517a81b235f35fae2128af64393371c2fdde1c8f9a638e4a1a",
}
CONDITIONS = ("clean", "sym30", "sym40", "sym50", "asym30", "asym40", "asym50")
SEEDS = (71, 72, 73)
BUDGETS = (400, 800, 1200, 1600, 2000)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(args):
    bank = args.bank.resolve()
    groups_file = args.superclasses.resolve()
    groups = list(json.loads(groups_file.read_text()).values())
    if len(groups) != 20 or any(len(group) != 5 for group in groups):
        raise ValueError("Expected 20 CIFAR-100 superclasses of five classes")
    if sorted(sum(groups, [])) != list(range(100)):
        raise ValueError("CIFAR-100 superclasses must partition indices 0 through 99")
    for name, expected in BANK_HASHES.items():
        if file_sha256(bank / f"{name}.npy") != expected:
            raise ValueError(f"BYOL checksum mismatch for {name}")
    out = args.output.resolve()
    out.mkdir(parents=True)
    for name in BANK_HASHES:
        shutil.copyfile(bank / f"{name}.npy", out / f"{name}.npy")
    shutil.copyfile(groups_file, out / "superclasses.json")
    write_json(out / "inputs.json", {
        "bank": str(bank),
        "bank_sha256": BANK_HASHES,
        "superclasses_source": str(groups_file),
        "superclasses_sha256": file_sha256(out / "superclasses.json"),
        "noise_recipe": "lab synthetic pools: exact uniformly wrong symmetric subset; "
                        "per-row superclass-cycle asymmetric channel",
    })
    print(f"Prepared all 50,000 train rows and 10,000 test rows in {out}", flush=True)


def read_inputs(root):
    meta = json.loads((root / "inputs.json").read_text())
    if meta["bank_sha256"] != BANK_HASHES:
        raise ValueError("Unexpected prepared bank manifest")
    for name, expected in BANK_HASHES.items():
        if file_sha256(root / f"{name}.npy") != expected:
            raise ValueError(f"Prepared input corrupted: {name}")
    if file_sha256(root / "superclasses.json") != meta["superclasses_sha256"]:
        raise ValueError("Prepared superclass file checksum mismatch")
    groups = list(json.loads((root / "superclasses.json").read_text()).values())
    successor = np.empty(100, dtype=np.int64)
    for group in groups:
        successor[group] = np.roll(group, -1)
    return meta, successor


def noise_spec(condition):
    if condition == "clean":
        return "sym", 0.0
    return condition[:-2], int(condition[-2:]) / 100


def observed_labels(clean, condition, seed, successor):
    """Use the same condition-seed noise pools as the lab's other BYOL studies."""
    kind, rate = noise_spec(condition)
    corrupted = clean.copy()
    rng = np.random.default_rng(seed)
    if kind == "sym":
        sample = rng.choice(len(clean), int(rate * len(clean)), replace=False)
        draws = rng.integers(99, size=len(sample))
        corrupted[sample] = draws + (draws >= clean[sample])
    else:
        shifted = successor[clean]
        stay = np.float64(np.float32(1.0 - rate))
        stay /= stay + np.float64(np.float32(rate))
        cutoff = np.where(clean < shifted, stay, 1.0 - stay)
        first = rng.random(len(clean)) < cutoff
        corrupted = np.where(first, np.minimum(clean, shifted), np.maximum(clean, shifted))
    return corrupted


class FeatureDataset:
    """Expose the official feature-only dataset sample contracts without CIFAR downloads."""

    def __init__(self, inputs, successor, condition, seed, is_train):
        split = "train" if is_train else "test"
        n_rows = 50000 if is_train else 10000
        self.train = is_train
        self.features = np.load(inputs / f"{split}_features.npy", allow_pickle=False)
        self.targets = np.load(inputs / f"{split}_targets.npy", allow_pickle=False).astype(np.int64)
        if self.features.shape != (n_rows, 512) or self.targets.shape != (n_rows,):
            raise ValueError(f"Unexpected {split} dimensions")
        if self.features.dtype != np.float32:
            raise ValueError(f"Expected raw float32 BYOL features for {split}")
        self.classes = list(range(100))
        self.only_features = True
        if self.train:
            self.noisy_labels = observed_labels(self.targets, condition, seed, successor)
            self.is_noisy = self.noisy_labels != self.targets
            self.noise_rate = noise_spec(condition)[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.train:
            return self.features[index], self.noisy_labels[index], self.targets[index], index
        return self.features[index], self.targets[index], index


def load_upstream(upstream):
    deep_al = upstream.resolve() / "deep-al"
    files = sorted([*deep_al.glob("pycls/**/*.py"), *deep_al.glob("tools/*.py"), deep_al / CONFIG])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(deep_al)).encode() + b"\0" + bytes.fromhex(file_sha256(path)))
    if digest.hexdigest() != UPSTREAM_SOURCE_SHA256:
        raise ValueError(f"Expected unchanged official source at commit {UPSTREAM_COMMIT}: {deep_al}")
    os.chdir(deep_al / "tools")
    sys.path.insert(0, str(deep_al / "tools"))
    import train_al
    from pycls.datasets.utils.features import DATASET_FEATURES_DICT
    from pycls.lnl.AUM import AUMCalculator
    import yaml

    if not train_al.torch.cuda.is_available():
        raise RuntimeError("Pinned NPC, AUM and Framework 2 require CUDA")
    print(f"Using Netta NPC commit {UPSTREAM_COMMIT} and AUM calculator "
          f"{AUMCalculator.__module__}.{AUMCalculator.__name__}", flush=True)
    return train_al, DATASET_FEATURES_DICT, deep_al / CONFIG, yaml


def configure_data(official, feature_paths, inputs, successor, condition, seed):
    for split in ("train", "test"):
        feature_paths[split]["CIFAR100"]["byol"] = str(inputs / f"{split}_features.npy")

    class ByolData(official.Data):
        def getDataset(self, save_dir, isTrain=True, isDownload=False, only_features=False,
                       representation_model="simclr", exp_dir=None, project_features=False):
            if not only_features or representation_model != "byol" or project_features:
                raise ValueError("Expected the stock raw fixed-feature linear classifier path")
            dataset = FeatureDataset(inputs, successor, condition, seed, isTrain)
            return dataset, len(dataset)

    official.Data = ByolData


def save_rng(torch):
    algorithm, keys, position, has_gauss, gauss = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), position, has_gauss, gauss),
    }


def restore_rng(torch, state):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, gauss = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, gauss))


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(test_model, test_loader, cfg, episode, model, checkpoint_file=None):
    """Record index-aligned predictions from the unmodified official evaluator."""
    predictions = np.full(len(test_loader.dataset), -1, dtype="<i8")
    current_indices = None

    class IndexedBatches:
        dataset = test_loader.dataset

        def __len__(self):
            return len(test_loader)

        def __iter__(self):
            nonlocal current_indices
            for inputs, labels, indices in test_loader:
                current_indices = indices.numpy()
                yield inputs, labels, indices

    def record(_model, _inputs, logits):
        predictions[current_indices] = logits.detach().topk(5, dim=1).indices[:, 0].cpu().numpy()

    handle = model.register_forward_hook(record)
    try:
        accuracy = test_model(IndexedBatches(), cfg, episode, model, checkpoint_file)
    finally:
        handle.remove()
    if (predictions < 0).any():
        raise RuntimeError("Official evaluator did not visit every held-out row")
    targets = np.asarray(test_loader.dataset.targets)
    exact_accuracy = 100.0 * np.count_nonzero(predictions == targets) / len(targets)
    if not np.isclose(exact_accuracy, accuracy, rtol=0, atol=1e-5):
        raise RuntimeError("Recorded held-out predictions disagree with official accuracy")
    return exact_accuracy, array_sha256(predictions)


def install_snapshot_hooks(official, output, budgets):
    torch = official.torch
    stock_filter = official.LearningWithNoisyLabels.identify_noisy_samples
    stock_loader = official.Data.getIndexesDataLoader
    stock_train = official.train_model
    stock_test = official.test_model
    pending = None
    summaries = []

    def capture_filter(self, l_set, u_set, return_scores=False, silent=False):
        nonlocal pending
        result = stock_filter(self, l_set, u_set, return_scores=return_scores, silent=silent)
        if return_scores:
            keep, scores = result[:2]
            if not np.any(keep):
                raise RuntimeError(f"Official final AUM kept no samples at budget {len(l_set)}")
            if len(l_set) in budgets:
                pending = {
                    "acquired_indices": l_set.copy(),
                    "observed_labels": self.train_data.noisy_labels[l_set].copy(),
                    "aum_scores": np.asarray(scores).copy(),
                    "base_keep": keep.copy(),
                }
        return result

    def capture_loader(self, indexes, batch_size, data):
        if pending is not None:
            np.testing.assert_array_equal(indexes, pending["acquired_indices"][pending["base_keep"]])
            pending["rng"] = save_rng(torch)
        return stock_loader(self, indexes, batch_size, data)

    def capture_initial_fit(loader, model, optimizer, cfg, train_set):
        if pending is not None:
            budget = len(pending["acquired_indices"])
            directory = output / "fits" / f"{budget:05d}"
            directory.mkdir(parents=True)
            (directory / "fit_cfg.yaml").write_text(cfg.dump())
            rng = pending.pop("rng")
            np.savez(directory / "fit_inputs.npz", **pending)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng},
                       directory / "fit_state.pt")
        return stock_train(loader, model, optimizer, cfg, train_set)

    def capture_test(test_loader, cfg, episode, model, checkpoint_file=None):
        nonlocal pending
        if pending is None:
            return stock_test(test_loader, cfg, episode, model, checkpoint_file)
        accuracy, prediction_sha = evaluate(stock_test, test_loader, cfg, episode, model, checkpoint_file)
        budget = len(pending["acquired_indices"])
        result = {
            "budget": budget,
            "kept": int(pending["base_keep"].sum()),
            "accuracy": accuracy,
            "fitted_model_sha256": model_sha256(model),
            "prediction_sha256": prediction_sha,
        }
        directory = output / "fits" / f"{budget:05d}"
        result["fit_inputs_sha256"] = file_sha256(directory / "fit_inputs.npz")
        write_json(directory / "baseline_result.json", result)
        summaries.append(result)
        pending = None
        return accuracy

    official.LearningWithNoisyLabels.identify_noisy_samples = capture_filter
    official.Data.getIndexesDataLoader = capture_loader
    official.train_model = capture_initial_fit
    official.test_model = capture_test
    return summaries


def official_cli(official, args):
    """Execute the command-line setup in the unchanged upstream entry point."""
    sys.argv = [official.__file__, *args]
    source = ast.parse(Path(official.__file__).read_bytes())
    guard = source.body[-1]
    if not isinstance(guard, ast.If) or ast.unparse(guard.test) != "__name__ == '__main__'":
        raise RuntimeError("Pinned train_al.py has an unexpected command-line entry point")
    entry = ast.Module(body=guard.body, type_ignores=[])
    # YACS rejects an ndarray at assignment. Convert only the fixed-budget
    # reporting values, so the stock trainer receives the same numeric list.
    reporting = [
        node for node in ast.walk(entry)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
        and any(isinstance(target, ast.Attribute)
                and target.attr == "EXPECTED_CLEAN_SAMPLES_PER_CLASS"
                for target in node.targets)
    ]
    if len(reporting) != 1:
        raise RuntimeError("Unexpected fixed-budget reporting assignment in pinned CLI")
    node = reporting[0]
    node.value = ast.Call(ast.Attribute(node.value, "tolist", ast.Load()), [], [])
    exec(compile(ast.fix_missing_locations(entry), official.__file__, "exec"), vars(official))


def run(args):
    inputs = args.inputs.resolve()
    output = args.output.resolve()
    meta, successor = read_inputs(inputs)
    official, feature_paths, stock_config, yaml = load_upstream(args.upstream)
    configure_data(official, feature_paths, inputs, successor, args.condition, args.seed)

    kind, rate = noise_spec(args.condition)
    output.mkdir(parents=True)
    config = yaml.safe_load(stock_config.read_text())
    config["DATASET"]["REPRESENTATION_MODEL"] = "byol"
    config["DATASET"]["ROOT_DIR"] = str(inputs)
    config["NOISE"]["NOISE_TYPE"] = kind
    config["OUT_DIR"] = str(output)
    config_path = output / "byol.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cli = [
        "--cfg", str(config_path), "--exp-name", "official", "--seed", str(args.seed),
        "--al", "npc", "--initial_budget_size", "0", "--budget", "100",
        "--num_episodes", "2" if args.smoke else "20", "--distance_metric", "l2",
        "--features_normalization", "unit_norm", "--initial_delta", "0.65",
        "--delta_policy", "constant", "--lnl", "aum", "--noise_type", kind,
        "--noise_rate", str(rate), "--noise_mom_coeff", "0", "--use_linear_model",
        "--max_epoch", "2" if args.smoke else "500",
    ]
    provenance = {
        "condition": args.condition,
        "seed": args.seed,
        "smoke": args.smoke,
        "input_manifest": meta,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "wrapper_sha256": file_sha256(Path(__file__).resolve()),
        "official_cli": cli,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "numpy", "scipy", "pandas", "scikit-learn", "yacs", "aum")},
        "gpu": official.torch.cuda.get_device_name(),
    }
    print(json.dumps(provenance, sort_keys=True), flush=True)
    write_json(output / "provenance.json", provenance)
    budgets = (100, 200) if args.smoke else BUDGETS
    summaries = install_snapshot_hooks(official, output, budgets)
    official_cli(official, cli)
    if tuple(row["budget"] for row in summaries) != budgets:
        raise RuntimeError("Official run did not deliver every requested budget")
    write_json(output / "summary.json", {
        "condition": args.condition, "seed": args.seed, "smoke": args.smoke, "cells": summaries,
    })


def replay_fit(official, datasets, cfg, state, acquired, keep):
    data = official.Data(cfg)
    train_data, test_data = datasets
    test_loader = data.getTestLoader(test_data, cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    model = official.model_builder.build_model(cfg)
    optimizer = official.optim.construct_optimizer(cfg, model)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    retain = acquired[keep]
    restore_rng(official.torch, state["rng"])
    train_loader = data.getIndexesDataLoader(retain, cfg.TRAIN.BATCH_SIZE, train_data)
    loss = official.train_model(train_loader, model, optimizer, cfg, retain)
    episode = len(acquired) // cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[0] - 1
    accuracy, prediction_sha = evaluate(official.test_model, test_loader, cfg, episode, model)
    return {
        "kept": len(retain),
        "accuracy": accuracy,
        "fitted_model_sha256": model_sha256(model),
        "prediction_sha256": prediction_sha,
        "train_loss": float(loss),
    }


def load_mask_function(path, name):
    spec = importlib.util.spec_from_file_location("matched_retention", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name)
    if not callable(function):
        raise ValueError("The requested mask function must be callable")
    return function


def fit(args):
    snapshot = args.snapshot.resolve()
    output = args.output.resolve()
    inputs = args.inputs.resolve()
    run_meta = json.loads((snapshot.parents[1] / "provenance.json").read_text())
    baseline = json.loads((snapshot / "baseline_result.json").read_text())
    condition, seed = run_meta["condition"], run_meta["seed"]
    meta, successor = read_inputs(inputs)
    if meta["superclasses_sha256"] != run_meta["input_manifest"]["superclasses_sha256"]:
        raise ValueError("The prepared noise recipe differs from the baseline run")
    fit_input_file = snapshot / "fit_inputs.npz"
    fit_inputs_sha = file_sha256(fit_input_file)
    if fit_inputs_sha != baseline["fit_inputs_sha256"]:
        raise ValueError("The frozen acquisition and AUM snapshot has changed")
    with np.load(fit_input_file, allow_pickle=False) as payload:
        acquired = payload["acquired_indices"]
        observed = payload["observed_labels"]
        scores = payload["aum_scores"]
        base_keep = payload["base_keep"]
    budget = len(acquired)
    if budget != baseline["budget"] or int(base_keep.sum()) != baseline["kept"]:
        raise ValueError("The baseline snapshot disagrees with its result")
    if args.mask_function:
        filename, name = args.mask_function.rsplit(":", 1)
        mask_path = Path(filename).resolve()

    official, feature_paths, _, _ = load_upstream(args.upstream)
    configure_data(official, feature_paths, inputs, successor, condition, seed)
    from yacs.config import CfgNode

    output.mkdir(parents=True)
    cfg = official.cfg
    with (snapshot / "fit_cfg.yaml").open() as stream:
        cfg.clear()
        cfg.update(CfgNode.load_cfg(stream))
    cfg.DATASET.ROOT_DIR = str(inputs)
    cfg.OUT_DIR = cfg.EXP_DIR = cfg.EPISODE_DIR = str(output)
    official.lu.setup_logging(cfg)
    data = official.Data(cfg)
    datasets = tuple(data.getDataset(inputs, isTrain=is_train, only_features=True,
                                     representation_model="byol")[0] for is_train in (True, False))
    np.testing.assert_array_equal(datasets[0].noisy_labels[acquired], observed)
    state = official.torch.load(snapshot / "fit_state.pt", map_location="cpu", weights_only=True)

    identity = replay_fit(official, datasets, cfg, state, acquired, base_keep)
    result_meta = {
        "budget": budget,
        "condition": condition,
        "seed": seed,
        "fit_inputs_sha256": fit_inputs_sha,
        "baseline_prediction_sha256": baseline["prediction_sha256"],
        "baseline_fitted_model_sha256": baseline["fitted_model_sha256"],
    }
    identity.update(result_meta, mask_function=None)
    write_json(output / "identity_result.json", identity)
    if (identity["prediction_sha256"] != baseline["prediction_sha256"]
            or identity["fitted_model_sha256"] != baseline["fitted_model_sha256"]
            or not np.isclose(identity["accuracy"], baseline["accuracy"], rtol=0, atol=1e-5)):
        raise RuntimeError("Identity fit diverged from the baseline model or held-out predictions. "
                           "Inspect identity_result.json and match GPU/runtime before trying candidates.")
    if not args.mask_function:
        write_json(output / "result.json", identity)
        return

    mask_function = load_mask_function(mask_path, name)
    fit_features = datasets[0].features[acquired]
    candidate = np.asarray(mask_function(fit_features, observed.copy(), scores.copy(), base_keep.copy()))
    if candidate.shape != base_keep.shape or candidate.dtype != np.bool_:
        raise ValueError("The mask function must return a boolean vector aligned to the acquired rows")
    if not np.array_equal(np.bincount(observed[candidate], minlength=100),
                          np.bincount(observed[base_keep], minlength=100)):
        raise ValueError("The mask function must preserve the retained count in every observed class")
    result = replay_fit(official, datasets, cfg, state, acquired, candidate)
    result.update(
        result_meta,
        mask_function=args.mask_function,
        mask_function_sha256=file_sha256(mask_path),
        changed_pairs=int((base_keep & ~candidate).sum()),
        baseline_retained_clean=int((base_keep & ~datasets[0].is_noisy[acquired]).sum()),
        retained_clean=int((candidate & ~datasets[0].is_noisy[acquired]).sum()),
    )
    np.save(output / "keep.npy", candidate, allow_pickle=False)
    write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare", help="verify and stage the frozen bank")
    for flag in ("bank", "superclasses", "output"):
        prep.add_argument(f"--{flag}", type=Path, required=True)
    prep.set_defaults(handler=prepare)

    baseline = actions.add_parser("run", help="run all official NPC episodes on BYOL")
    for flag in ("upstream", "inputs", "output"):
        baseline.add_argument(f"--{flag}", type=Path, required=True)
    baseline.add_argument("--condition", choices=CONDITIONS, required=True)
    baseline.add_argument("--seed", type=int, choices=SEEDS, required=True)
    baseline.add_argument("--smoke", action="store_true")
    baseline.set_defaults(handler=run)

    replay = actions.add_parser("fit", help="verify and optionally replace the pinned final AUM mask")
    for flag in ("upstream", "inputs", "snapshot", "output"):
        replay.add_argument(f"--{flag}", type=Path, required=True)
    replay.add_argument("--mask-function", metavar="FILE:CALLABLE")
    replay.set_defaults(handler=fit)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

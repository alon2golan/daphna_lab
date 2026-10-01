import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch

from deep_al.vast.data import load_manifest, select_labeled
from deep_al.vast.graph import build_kernel, cached_kernel, neighbor_indices, normalize_features, union_graph
from deep_al.vast.inference import (UnresolvedPropagationError, require_propagation, seed_evidence,
                                    propagate_literal, prepare_targets, mode_threshold)
from deep_al.vast.run import main, grid_commands
from deep_al.vast.training import balanced_batches, distillation_loss, predict_checkpoint
from deep_al.vast.upstream import _apply_degree_normalization_sparse, MLPClassifierTorch


def make_manifest(tmp_path, dataset="cifar100", per_class=4):
    classes = 200 if dataset == "tinyimagenet" else 100
    rng = np.random.RandomState(10)
    centers = rng.normal(size=(classes, 8))
    train_labels = np.repeat(np.arange(classes), per_class)
    train = centers[train_labels] + rng.normal(scale=0.01, size=(len(train_labels), 8))
    test_labels = np.arange(classes)
    spec = {"dataset": dataset, "representation": {"name": "synthetic-test", "source": "synthetic"},
            "alignment": "Generated features, labels, and IDs jointly in the same loop.", "class_ids": "classes.json"}
    (tmp_path / "classes.json").write_text(json.dumps([f"class-{i}" for i in range(classes)]))
    for split, features, labels in (("train", train, train_labels), ("test", centers, test_labels)):
        spec[split] = {}
        for kind, values in (("features", features.astype(np.float32)), ("labels", labels),
                             ("sample_ids", np.array([f"{split}-{i}" for i in range(len(labels))]))):
            name = f"{split}-{kind}.npy"
            np.save(tmp_path / name, values)
            spec[split][kind] = name
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path


def test_manifest_alignment_budget_and_metadata(tmp_path):
    manifest = make_manifest(tmp_path)
    data = load_manifest(manifest, 4)
    assert data.train_features.shape == (400, 8)
    assert len(data.provenance["sha256"]) == 7
    picked = select_labeled(data.train_labels, 100, 2, 23)
    np.testing.assert_array_equal(np.bincount(data.train_labels[picked]), np.full(100, 2))
    np.testing.assert_array_equal(picked, select_labeled(data.train_labels, 100, 2, 23))
    assert not np.array_equal(picked, select_labeled(data.train_labels, 100, 2, 24))


@pytest.mark.parametrize("problem", ["order", "overlap", "zero", "nan", "width", "labels", "ids", "budget"])
def test_manifest_rejects_invalid_inputs(tmp_path, problem):
    manifest = make_manifest(tmp_path)
    spec = json.loads(manifest.read_text())
    if problem == "order":
        del spec["alignment"]
        manifest.write_text(json.dumps(spec))
    elif problem == "overlap":
        np.save(tmp_path / "test-sample_ids.npy", np.array([f"train-{i}" for i in range(100)]))
    elif problem in ("zero", "nan"):
        x = np.load(tmp_path / "train-features.npy")
        x[0] = 0 if problem == "zero" else np.nan
        np.save(tmp_path / "train-features.npy", x)
    elif problem == "width":
        np.save(tmp_path / "test-features.npy", np.ones((100, 9)))
    elif problem == "labels":
        np.save(tmp_path / "train-labels.npy", np.full(400, 100))
    elif problem == "ids":
        np.save(tmp_path / "train-sample_ids.npy", np.array(["repeated"] * 400))
    with pytest.raises(ValueError):
        load_manifest(manifest, 5 if problem == "budget" else 4)


def test_author_degree_normalization_hand_reference():
    raw = sp.csr_matrix([[0.0, 0.5, 0.25], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    actual = _apply_degree_normalization_sparse(raw, 1.0, "log").toarray()
    expected = np.zeros((3, 3))
    expected[0, 1] = expected[1, 0] = 0.5 / np.sqrt(np.log1p(0.75 * 0.5))
    expected[0, 2] = expected[2, 0] = 0.25 / np.sqrt(np.log1p(0.75 * 0.25))
    np.testing.assert_allclose(actual, expected)
    single_edge = _apply_degree_normalization_sparse(sp.csr_matrix([[0., .5], [.5, 0.]]), 1.0, "log")
    assert single_edge[0, 1] > 1  # Do not clip the corrected kernel to 1.


def test_graph_smallest_connected_k_and_kernel():
    x = np.array([[1, 0], [1, .01], [-1, 0], [-1, .01]], dtype=np.float32)
    kernel, info = build_kernel(x, "dense")
    assert info["k"] == 2
    assert not kernel.diagonal().any()
    np.testing.assert_allclose(kernel.toarray(), kernel.T.toarray())
    z = normalize_features(x)
    neighbors = neighbor_indices(z, 3, "dense")
    assert connected_components(union_graph(neighbors, 1), directed=False, return_labels=False) == 2
    mask = union_graph(neighbors, 2).toarray()
    d2 = ((z[:, None].astype(float) - z[None, :]) ** 2).sum(axis=2)
    raw = np.exp(-d2) * mask
    degree = raw.sum(axis=1)
    expected = np.divide(raw, np.sqrt(np.log1p(degree[:, None] * degree[None, :])),
                         out=np.zeros_like(raw), where=mask > 0)
    np.testing.assert_allclose(kernel.toarray(), expected)


def test_faiss_matches_dense_and_removes_duplicate_self():
    if importlib.util.find_spec("faiss") is None:
        pytest.skip("FAISS not installed")
    z = normalize_features(np.random.RandomState(9).normal(size=(30, 8)))
    np.testing.assert_array_equal(neighbor_indices(z, 5, "faiss"), neighbor_indices(z, 5, "dense"))
    duplicate = np.tile(np.array([[1, 0]], dtype=np.float32), (6, 1))
    neighbors = neighbor_indices(duplicate, 2, "faiss")
    assert all(i not in neighbors[i] for i in range(len(duplicate)))


def test_graph_cap_and_cache_invalidation(tmp_path):
    x = np.array([[1, 0], [1, .01], [-1, 0], [-1, .01]], dtype=np.float32)
    with pytest.raises(ValueError, match="disconnected"):
        build_kernel(x, "dense", 1)
    first, meta = cached_kernel(x, "dense", 3, tmp_path)
    second, again = cached_kernel(x, "dense", 3, tmp_path)
    assert not meta["cache_hit"] and again["cache_hit"]
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    changed = x.copy()
    changed[0, 1] += 0.1
    _, different = cached_kernel(changed, "dense", 3, tmp_path)
    assert different["cache_key"] != meta["cache_key"]


def test_seed_evidence_is_sum_plus_prior():
    kernel = sp.csr_matrix([[0, .2, .4], [.2, 0, .3], [.4, .3, 0]])
    actual = seed_evidence(kernel, np.array([0, 1]), np.array([0, 0]), 2)
    np.testing.assert_allclose(actual, [[.7, .5], [.7, .5], [1.2, .5]])


def test_literal_propagation_synchronous_and_frozen_rows():
    kernel = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    evidence = np.array([[9., 1.], [1., 1.], [2., 1.]])
    result, info = propagate_literal(kernel, evidence, np.array([True, False, False]))
    # Round 1: row1 <- row1 + row0 + row2; row2 gets nothing (row1 was uniform).
    # Row1 freezes at 12/15=.8. Round2: row2 receives frozen row1, reaches14/18.
    np.testing.assert_allclose(result, [[9, 1], [12, 3], [14, 4]])
    assert info["rounds"] == 3  # Includes the final no-new-freeze round.
    assert [r["newly_frozen"] for r in info["history"]] == [1, 1, 0]
    with pytest.raises(RuntimeError, match="cap"):
        propagate_literal(kernel, evidence, np.array([True, False, False]), max_rounds=1)


@pytest.mark.parametrize("classes", [100, 200])
@pytest.mark.parametrize("per_class", [1, 2])
def test_literal_bound_and_one_round_stop(classes, per_class):
    n = classes * per_class + 5
    raw = sp.diags([np.full(n - 1, .8), np.full(n - 1, .8)], [-1, 1], shape=(n, n), format="csr")
    kernel = _apply_degree_normalization_sparse(raw, 1.0, "log")
    labeled = np.arange(classes * per_class)
    labels = np.repeat(np.arange(classes), per_class)
    mask = np.arange(n) < len(labeled)
    initial = seed_evidence(kernel, labeled, labels, classes)
    result, info = propagate_literal(kernel, initial, mask)
    bound = (1 / classes + per_class / np.sqrt(np.log(2))) / (1 + per_class / np.sqrt(np.log(2)))
    assert (result / result.sum(axis=1, keepdims=True)).max() <= bound + 1e-12
    assert info["rounds"] == 1 and info["history"][0]["newly_frozen"] == 0
    np.testing.assert_array_equal(result[mask], initial[mask])


def test_no_silent_propagation_default():
    with pytest.raises(UnresolvedPropagationError):
        require_propagation("unresolved")
    require_propagation("literal-paper")


def test_histogram_golden_boundary_modes_and_fallback():
    value, bimodal = mode_threshold(np.r_[np.full(100, .1), np.full(50, .9)])
    assert bimodal and value == pytest.approx(.884)
    assert mode_threshold(np.full(20, .9)) == (.75, False)
    assert mode_threshold(np.array([])) == (.75, False)
    # A tiny second bump does not qualify against the dominant peak.
    assert mode_threshold(np.r_[np.full(100, .1), [.9]]) == (.75, False)


def test_gate_top_class_statistic_chance_floor_and_labeled_exclusion(monkeypatch):
    monkeypatch.setattr("deep_al.vast.inference.mode_threshold", lambda _: (.75, False))
    evidence = np.array([[.95, .05], [.9, .1], [.6, .4], [.2, .8]])
    probs, weights, admitted, info = prepare_targets(evidence, np.array([True, False, False, False]))
    np.testing.assert_allclose(probs, evidence)
    np.testing.assert_allclose(info["class_thresholds"], [.75, .5])
    np.testing.assert_array_equal(admitted, [False, True, False, True])
    np.testing.assert_allclose(weights, [.95, .9, .6, .8])
    _, _, admitted, _ = prepare_targets(np.ones((4, 2)), np.zeros(4, dtype=bool))
    assert not admitted.any()  # Exactly chance is excluded.


def test_reverse_kl_and_sample_weighting_have_correct_gradient():
    p = torch.tensor([[.9, .1], [.4, .6]], dtype=torch.float64)
    q = torch.tensor([[.6, .4], [.8, .2]], dtype=torch.float64)
    logits = p.log().detach().requires_grad_()
    weights = torch.tensor([.6, .8], dtype=torch.float64)
    actual = distillation_loss(logits, q, weights)
    per_sample = (p * (p.log() - q.log())).sum(dim=1)
    expected = (per_sample * weights).mean()
    torch.testing.assert_close(actual, expected)
    assert abs(actual.item() - (per_sample.mean() * weights.mean()).item()) > .001
    assert abs(actual.item() - ((q * (q.log() - p.log())).sum(dim=1) * weights).mean().item()) > .001
    actual.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda x: distillation_loss(x, q, weights), (logits,))
    assert distillation_loss(logits[:0], q[:0], weights[:0]).item() == 0


def test_balanced_sampler_covers_pool_and_cycles_labels():
    labeled, admitted = np.array([0, 1]), np.arange(2, 103)
    batches = list(balanced_batches(labeled, admitted, 16, np.random.RandomState(7)))
    actual = []
    for indices, mask in batches:
        assert mask.sum() == 2
        assert np.isin(indices[mask], labeled).all()
        actual.extend(indices[~mask].tolist())
    assert sorted(actual) == admitted.tolist()
    replay = list(balanced_batches(labeled, admitted, 16, np.random.RandomState(7)))
    for (x, m), (y, n) in zip(batches, replay):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(m, n)
    only = list(balanced_batches(labeled, [], 16, np.random.RandomState(7)))
    assert len(only) == 1 and only[0][1].all()


def test_mlp_author_architecture():
    model = MLPClassifierTorch(8, 256, 100)
    assert model(torch.ones(4, 8)).shape == (4, 100)
    assert [type(layer) for layer in model.network] == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]


def test_cli_preflight_and_unresolved_no_outputs(tmp_path):
    manifest = make_manifest(tmp_path)
    main(["--manifest", str(manifest), "--labels-per-class", "1", "--preflight"])
    output = tmp_path / "blocked"
    with pytest.raises(SystemExit) as exc:
        main(["--manifest", str(manifest), "--labels-per-class", "1", "--output-dir", str(output)])
    assert exc.value.code == 2 and not output.exists()


def test_grid_has_nine_settings_one_seed(tmp_path):
    paths = []
    for dataset in ("cifar100", "imagenet100", "tinyimagenet"):
        folder = tmp_path / dataset
        folder.mkdir()
        paths.append(make_manifest(folder, dataset))
    commands = grid_commands(paths, tmp_path / "runs", seed=41, device="cuda")
    assert {c[c.index("--device") + 1] for c in commands} == {"cuda"}
    assert len(commands) == 9
    assert {c[c.index("--seed") + 1] for c in commands} == {"41"}
    assert len({c[-1] for c in commands}) == 9
    with pytest.raises(ValueError):
        grid_commands(paths[:2], tmp_path / "runs")


def test_end_to_end_repeatability_final_checkpoint_and_test_isolation(tmp_path):
    manifest = make_manifest(tmp_path)
    common = ["--manifest", str(manifest), "--labels-per-class", "1", "--seed", "11",
              "--backend", "dense", "--device", "cpu", "--epochs", "2",
              "--propagation", "literal-paper", "--cache-dir", str(tmp_path / "cache")]
    outputs = [tmp_path / "one", tmp_path / "two"]
    main(common + ["--output-dir", str(outputs[0])])
    # Poison test labels between runs. It must not change graph, targets or weights.
    test_labels = np.load(tmp_path / "test-labels.npy")
    np.save(tmp_path / "test-labels.npy", (test_labels + 1) % 100)
    main(common + ["--output-dir", str(outputs[1])])
    one = torch.load(outputs[0] / "final.pt", weights_only=True)
    two = torch.load(outputs[1] / "final.pt", weights_only=True)
    assert one["epoch"] == two["epoch"] == 2
    for key in one["state_dict"]:
        torch.testing.assert_close(one["state_dict"][key], two["state_dict"][key], atol=0, rtol=0)
    data = load_manifest(manifest)
    predictions = predict_checkpoint(outputs[0] / "final.pt", data.test_features)
    np.testing.assert_array_equal(predictions, np.load(outputs[0] / "test_predictions.npy"))
    np.testing.assert_array_equal(predictions, np.load(outputs[1] / "test_predictions.npy"))
    result = json.loads((outputs[0] / "result.json").read_text())
    assert result["seed_count"] == 1 and result["sem"] is None and result["sd"] is None
    assert result["confirmed_reproduction"] is False
    assert json.loads((outputs[1] / "graph.json").read_text())["cache_hit"]
    assert len((outputs[0] / "training.jsonl").read_text().splitlines()) == 2
    with pytest.raises(SystemExit):
        main(common + ["--output-dir", str(outputs[0])])


def test_entrypoint_from_different_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools" / "train_vast.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--seed" in result.stdout

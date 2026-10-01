"""Optional UI checks: install requirements-viewer.txt to exercise these."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deep_al.vast.inference import propagate_literal
from deep_al.vast.trace import PropagationTrace


def test_viewer_slider_roles_filters_and_missing_trace(tmp_path):
    app = Path(__file__).resolve().parents[2] / "tools/view_vast.py"
    at = AppTest.from_file(str(app), default_timeout=20).run()
    at.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception
    assert "no propagation replay logs" in at.warning[0].value
    trace = PropagationTrace(tmp_path / "propagation_trace", np.array([[1., 0], [1., 1], [0., 1]]),
                             np.array(["a", "b", "c"]), np.array([True, False, False]), ["A", "B"])
    propagate_literal(sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float),
                      np.array([[9., 1.], [1., 1.], [2., 1.]]),
                      np.array([True, False, False]), observer=trace)
    at.run()
    assert not at.exception
    initial_plot = json.loads(at.get("plotly_chart")[0].proto.spec)
    at.slider[0].set_value(1).run()
    assert not at.exception
    assert [metric.value for metric in at.metric] == ["2", "2", "1", "2"]
    at.sidebar.selectbox[0].select("Newly frozen").run()
    assert list(at.dataframe[0].value["sample_id"]) == ["b"]
    for mode in ("Predicted class", "Confidence"):
        at.sidebar.selectbox[1].select(mode).run()
        assert not at.exception
    at.slider[0].set_value(2).run()
    assert list(at.dataframe[0].value["sample_id"]) == ["c"]
    at.slider[0].set_value(3).run()
    assert not at.exception and len(at.dataframe[0].value) == 0
    final_plot = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert initial_plot["layout"]["xaxis"]["range"] == final_plot["layout"]["xaxis"]["range"]
    assert initial_plot["layout"]["yaxis"]["range"] == final_plot["layout"]["yaxis"]["range"]


def test_viewer_selects_runs_in_dataset_output_root(tmp_path):
    app = Path(__file__).resolve().parents[2] / "tools/view_vast.py"
    for name, seed_mask in (("labels1", [True, False, False]),
                            ("labels2", [True, True, False])):
        trace = PropagationTrace(tmp_path / name / "propagation_trace",
                                 np.array([[1., 0], [1., 1], [0., 1]]),
                                 np.array(["a", "b", "c"]), np.array(seed_mask), ["A", "B"])
        mask = np.array(seed_mask)
        trace(0, np.full((3, 2), .5), np.zeros(3, bool), np.zeros(3, bool),
              mask, np.zeros(3, bool))
    at = AppTest.from_file(str(app), default_timeout=20).run()
    at.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == "labels1"
    assert at.metric[3].value == "1"
    at.sidebar.selectbox[0].select("labels2").run()
    assert not at.exception
    assert at.metric[3].value == "2"

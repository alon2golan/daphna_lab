"""Launch: python -m streamlit run deep_al/tools/view_vast.py -- --run-dir RUN"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(max_entries=8)
def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="VAST propagation", layout="wide")
    st.title("VAST propagation replay")
    run_path = st.sidebar.text_input("Run output directory", args.run_dir)
    if not run_path:
        st.info("Enter the output directory of a new VAST run (the folder containing result.json).")
        return
    root = Path(run_path).expanduser()
    trace = root / "propagation_trace"
    if not (trace / "metadata.json").is_file():
        st.warning("This run has no propagation replay logs. Run VAST again with the updated code; old summary logs cannot reconstruct sample roles.")
        return
    try:
        meta = json.loads((trace / "metadata.json").read_text())
        if meta["version"] != 1:
            st.error("Unsupported replay format.")
            return
        samples = load_arrays(str(trace / "samples.npz"))
        frames = meta["frames"]
        if not frames:
            st.info("The layout is ready; waiting for the first propagation snapshot.")
            return
        iteration = st.slider("Propagation iteration (0 = initialization)", 0, len(frames) - 1, 0) if len(frames) > 1 else 0
        frame = load_arrays(str(trace / frames[iteration]["file"]))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read replay: {exc}")
        return
    st.caption("Experimental literal-paper propagation. Sender and receiver roles are recorded before each update; confidence, predictions and frozen states are shown after it. A sample can both send and receive.")
    if iteration == 0:
        st.info("Initialization: no update has happened yet. Diamonds mark labeled seeds; frozen seeds are included in the frozen count.")
    elif len(frames) == 2 and not frame["newly_frozen"].any():
        st.info("Propagation stopped after one update with no new freezes. This is the known literal-paper behavior at the smaller label budgets.")
    columns = st.columns(4)
    for column, key, title in zip(columns, ("senders", "receivers", "newly_frozen", "frozen"),
                                  ("Senders", "Receivers", "Newly frozen", "All frozen (including seeds)")):
        column.metric(title, int(frame[key].sum()))
    n = len(samples["xy"])
    cap = int(st.sidebar.number_input("Maximum displayed samples", min_value=1, max_value=n, value=min(n, 20000)))
    indices = np.sort(np.random.RandomState(0).choice(n, cap, replace=False)) if cap < n else np.arange(n)
    role = np.array(["Neither"] * n, dtype="U24")
    role[frame["senders"]] = "Sender only"
    role[frame["receivers"]] = "Receiver only"
    role[frame["senders"] & frame["receivers"]] = "Sender + receiver"
    show = st.sidebar.selectbox("Show samples", ["All", "Senders", "Receivers", "Newly frozen", "Frozen", "Labeled seeds"])
    filters = {"Senders": frame["senders"], "Receivers": frame["receivers"],
               "Newly frozen": frame["newly_frozen"], "Frozen": frame["frozen"], "Labeled seeds": samples["labeled"]}
    if show != "All":
        # Filter first so rare newly frozen samples remain inspectable at any cap.
        candidates = np.flatnonzero(filters[show])
        indices = np.sort(np.random.RandomState(0).choice(candidates, cap, replace=False)) if len(candidates) > cap else candidates
    color_by = st.sidebar.selectbox("Color by", ["Propagation role", "Predicted class", "Confidence"])
    class_names = np.asarray(meta["class_ids"])[frame["predicted"]]
    symbols = np.where(samples["labeled"], "diamond", np.where(frame["newly_frozen"], "star", np.where(frame["frozen"], "square", "circle")))
    sizes = np.where(frame["newly_frozen"], 13, np.where(samples["labeled"], 9, 5))
    fig = go.Figure()
    groups = [(name, indices[role[indices] == name], color) for name, color in
              (("Neither", "#94a3b8"), ("Sender only", "#2563eb"), ("Receiver only", "#ea580c"), ("Sender + receiver", "#9333ea"))] if color_by == "Propagation role" else [(color_by, indices, None)]
    for name, rows, color in groups:
        marker = dict(size=sizes[rows], symbol=symbols[rows], opacity=0.8)
        if color is not None:
            marker["color"] = color
        else:
            marker.update(color=frame["confidence"][rows] if color_by == "Confidence" else frame["predicted"][rows],
                          colorscale="Viridis" if color_by == "Confidence" else "Turbo", showscale=True,
                          cmin=0, cmax=1 if color_by == "Confidence" else len(meta["class_ids"]) - 1,
                          colorbar=dict(title=color_by))
        custom = np.column_stack([samples["sample_ids"][rows], class_names[rows], frame["confidence"][rows],
                                  role[rows], frame["frozen"][rows], frame["newly_frozen"][rows], samples["labeled"][rows]])
        fig.add_trace(go.Scattergl(x=samples["xy"][rows, 0], y=samples["xy"][rows, 1], mode="markers",
                                  name=name, marker=marker, customdata=custom,
                                  hovertemplate="ID: %{customdata[0]}<br>Predicted: %{customdata[1]}<br>Confidence: %{customdata[2]}<br>%{customdata[3]}<br>Frozen: %{customdata[4]}<br>Newly frozen: %{customdata[5]}<br>Labeled seed: %{customdata[6]}<extra></extra>"))
    # Fix both axes across rounds and filters; retain user zoom while sliding.
    limits = []
    for axis in range(2):
        lo, hi = float(samples["xy"][:, axis].min()), float(samples["xy"][:, axis].max())
        pad = max((hi - lo) * .05, .01)
        limits.append([lo - pad, hi + pad])
    fig.update_layout(height=650, uirevision=str(root.resolve()), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="PC 1", range=limits[0]), yaxis=dict(title="PC 2", range=limits[1]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {len(indices):,} of {n:,} samples; counters include all samples. Circle = unfrozen, square = previously frozen, star = newly frozen, diamond = labeled seed. PCA is fixed across all iterations, fitted on {meta['projection']['fit_count']:,} normalized feature vectors. Predicted classes are propagation estimates, including for seeds.")
    with st.expander("Inspect displayed samples"):
        st.dataframe({"sample_id": samples["sample_ids"][indices], "predicted_class": class_names[indices],
                      "confidence": frame["confidence"][indices], "role": role[indices],
                      "frozen": frame["frozen"][indices], "newly_frozen": frame["newly_frozen"][indices],
                      "labeled_seed": samples["labeled"][indices]}, use_container_width=True)
    if (root / "result.json").is_file():
        result = json.loads((root / "result.json").read_text())
        st.caption(f"Training finished: {result['epochs']} epochs · seed {result['seed']} · test accuracy {result['test_accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()

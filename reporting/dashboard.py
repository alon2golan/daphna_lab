"""
Streamlit dashboard for Active Learning experiment results.

Run (from the repo root):
    streamlit run reporting/dashboard.py

Remote access from your laptop:
    ssh -L 8501:localhost:8501 <user>@<university-server>
    → open http://localhost:8501

The dashboard has five tabs:
  1. Sweeps       – overview of all sbatch/local sweeps with progress bars
  2. Compare      – seed-averaged comparison charts with std bands
  3. Run Details  – per-run metrics, parameters, and log viewer
  4. Export       – download a self-contained HTML report
  5. Launch       – configure and fire new sweeps from the browser
"""

import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reporting.html_export import generate_report
from reporting.scanner import (
    get_all_aggregated,
    get_sweeps_summary,
    load_all_results,
)
from deep_al.pycls.noise.nu_prime_cache import load_cache as _load_nu_cache

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AL Experiment Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 AL Dashboard")
    output_dir = st.text_input(
        "Output directory",
        value=str(REPO_ROOT / "output"),
        help="Root directory scanned recursively for result.json files.",
    )
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 30 s.")

# ── Data loading ──────────────────────────────────────────────────────────────

_STATUS_EMOJI = {"completed": "✅", "running": "🔄", "failed": "❌"}
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# Persisted launches log (survives page refresh)
_LAUNCHES_FILE = REPO_ROOT / "logs" / "sweep" / "dashboard_launches.json"
# Persisted hidden sweeps (survives page refresh)
_HIDDEN_SWEEPS_FILE = REPO_ROOT / "logs" / "sweep" / "dashboard_hidden_sweeps.json"


def _load_hidden_sweeps() -> set:
    if _HIDDEN_SWEEPS_FILE.exists():
        try:
            return set(json.loads(_HIDDEN_SWEEPS_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _save_hidden_sweeps(hidden: set) -> None:
    _HIDDEN_SWEEPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HIDDEN_SWEEPS_FILE.write_text(json.dumps(sorted(hidden), indent=2))


def _load_launches() -> list:
    if _LAUNCHES_FILE.exists():
        try:
            return json.loads(_LAUNCHES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_launches(launches: list) -> None:
    _LAUNCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAUNCHES_FILE.write_text(json.dumps(launches, indent=2, default=str))


def _is_process_running(pid) -> bool:
    """Return True if a process with this PID is currently alive."""
    if not pid or not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _import_swt():
    """Import sweep_train_al without requiring batch_utils/__init__.py."""
    spec = importlib.util.spec_from_file_location(
        "sweep_train_al",
        REPO_ROOT / "batch_utils" / "sweep_train_al.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(ttl=30, show_spinner="Scanning for results…")
def _load_results(directory: str) -> pd.DataFrame:
    return load_all_results(directory)


df = _load_results(output_dir)

if df.empty:
    st.warning(
        "No `result.json` files found under the output directory.  "
        "Switch to the **🚀 Launch** tab to start your first sweep!"
    )

# ── Helpers ───────────────────────────────────────────────────────────────────


def _badge(status: str) -> str:
    return f"{_STATUS_EMOJI.get(status, '❓')} {status}"


def _diff_labels(agg_list: list) -> list[str]:
    """
    Given a list of agg dicts, return a short label per entry that includes
    only the param=value parts that actually differ across the set.
    Falls back to the full config_label when everything is identical.
    """
    def _parse(label: str) -> dict:
        parts = {}
        for part in label.split(" | "):
            if "=" in part:
                k, v = part.split("=", 1)
                parts[k.strip()] = v.strip()
            else:
                parts[part.strip()] = True
        return parts

    parsed = [_parse(a["config_label"]) for a in agg_list]
    all_keys = {k for p in parsed for k in p}
    # keys whose value varies across configs
    diff_keys = [
        k for k in all_keys
        if len({p.get(k) for p in parsed}) > 1
    ]
    if not diff_keys:
        # nothing differs — just use sweep_name as disambiguator
        return [a["sweep_name"] for a in agg_list]
    # Preserve original key order from _LABEL_KEY_ORDER, then any extras
    from reporting.scanner import _LABEL_KEY_ORDER as _LKO
    ordered = [k for k in _LKO if k in diff_keys]
    ordered += sorted(k for k in diff_keys if k not in ordered)
    def _fmt(k: str, v: str) -> str:
        """Shorten well-known long values (e.g. strip cfg_file directory)."""
        if k == "cfg_file":
            return Path(v).stem  # e.g. "RESNET18" or "RESNET18_noisy"
        return v

    short = []
    for p in parsed:
        # Use "—" for keys that are absent in this config (e.g. oracle has no tau)
        parts = [f"{k}={_fmt(k, p[k]) if k in p else '—'}" for k in ordered]
        short.append(" | ".join(parts) if parts else "?")
    return short


def _build_comparison_chart(
    agg_list: list, x_axis: str = "Episode", short_labels: list[str] | None = None
) -> go.Figure:
    fig = go.Figure()
    for i, agg in enumerate(agg_list):
        if agg["agg"].empty:
            continue
        color = _COLORS[i % len(_COLORS)]
        agg_df = agg["agg"].reset_index(drop=True)

        x_col = "labeled_size" if x_axis == "Labeled Set Size" else "episode"
        x = list(agg_df[x_col])
        mean = list(agg_df["test_acc_mean"])
        std = list(agg_df["test_acc_std"])
        _short = short_labels[i] if short_labels else None
        label = _short or f"{agg['sweep_name']} / {agg['config_label']}"

        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines+markers",
                name=f"{label} (n={agg['n_seeds']})",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    + (
                        "Episode: %{x}"
                        if x_axis == "Episode"
                        else "Labeled: %{x}"
                    )
                    + "<br>Test Acc: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title="Test Accuracy (%)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(t=40),
    )
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_sweeps, tab_compare, tab_details, tab_nu, tab_export, tab_launch = st.tabs(
    ["📊 Sweeps", "📈 Compare", "🔍 Run Details", "📉 ν' Cache", "📤 Export", "🚀 Launch"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SWEEPS OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab_sweeps:
    st.header("Experiment Sweeps")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch tab.")
    else:
        summary = get_sweeps_summary(df)
        if summary.empty:
            st.info("No sweeps found.")
        else:
            _status_filter = st.multiselect(
                "Show sweep status",
                options=["🔄 running", "✅ completed", "❌ has failures"],
                default=["🔄 running", "✅ completed", "❌ has failures"],
                key="sweep_status_filter",
            )
            _last_sweep_id = st.session_state.get("last_launched_sweep_id")
            _hidden_sweeps = _load_hidden_sweeps()

            _frow1, _frow2 = st.columns([4, 2])
            _frow1.write("")  # spacer
            _show_hidden = _frow2.toggle(
                f"Show hidden ({len(_hidden_sweeps)})",
                value=False,
                key="show_hidden_sweeps",
                disabled=not _hidden_sweeps,
            )
            if _hidden_sweeps and _show_hidden:
                if _frow2.button("Restore all hidden", key="restore_all_hidden"):
                    _save_hidden_sweeps(set())
                    st.rerun()

            for _, row in summary.iterrows():
                if row["running"] > 0:
                    _row_status = "🔄 running"
                elif row["failed"] > 0:
                    _row_status = "❌ has failures"
                else:
                    _row_status = "✅ completed"
                if _row_status not in _status_filter:
                    continue
                _is_hidden = row["sweep_id"] in _hidden_sweeps
                if _is_hidden and not _show_hidden:
                    continue
                pct = row["completed"] / row["total"] if row["total"] > 0 else 0.0
                date_str = (
                    row["created_at"].strftime("%Y-%m-%d %H:%M")
                    if pd.notna(row["created_at"])
                    else "?"
                )
                header = (
                    f"**{row['sweep_name']}**  —  {row['total']} jobs "
                    f"| ✅ {row['completed']}  🔄 {row['running']}  ❌ {row['failed']}"
                    f"  |  {date_str}"
                )
                _is_highlighted = bool(_last_sweep_id and row["sweep_id"] == _last_sweep_id)
                if _is_highlighted:
                    st.success("🚀 Newly launched sweep")
                if _is_hidden:
                    header = f"~~{header}~~ *(hidden)*"
                with st.expander(header, expanded=_is_highlighted):
                    st.progress(pct, text=f"{row['completed']}/{row['total']} completed")
                    _meta_col, _btn_col = st.columns([5, 1])
                    _meta_col.markdown(
                        f"**Sweep ID:** `{row['sweep_id']}` &nbsp;|&nbsp; "
                        f"**Commit:** `{str(row['commit_hash'])[:8]}`"
                    )
                    if _is_hidden:
                        if _btn_col.button(
                            "👁 Restore",
                            key=f"restore_{row['sweep_id']}",
                        ):
                            _hidden_sweeps.discard(row["sweep_id"])
                            _save_hidden_sweeps(_hidden_sweeps)
                            st.rerun()
                    else:
                        if _btn_col.button(
                            "🙈 Hide",
                            key=f"hide_{row['sweep_id']}",
                            help="Hide this sweep from the dashboard (non-destructive, files are kept)",
                        ):
                            _hidden_sweeps.add(row["sweep_id"])
                            _save_hidden_sweeps(_hidden_sweeps)
                            st.rerun()

                    sweep_runs = df[df["sweep_id"] == row["sweep_id"]].copy()
                    show = [
                        c for c in [
                            "config_label", "p_seed", "status",
                            "total_episodes_done", "total_episodes_planned",
                            "final_test_acc",
                        ]
                        if c in sweep_runs.columns
                    ]
                    rename = {
                        "config_label": "Config",
                        "p_seed": "Seed",
                        "status": "Status",
                        "total_episodes_done": "Done",
                        "total_episodes_planned": "Planned",
                        "final_test_acc": "Final Test Acc",
                    }
                    st.dataframe(
                        sweep_runs[show].rename(columns=rename),
                        width="stretch",
                        hide_index=True,
                    )

                    st.divider()
                    _sid = row["sweep_id"]

                    # ── Per-run deletion ──────────────────────────────────────
                    with st.expander("🗑 Delete individual runs", expanded=False):
                        _run_labels = []
                        for _, _r in sweep_runs.iterrows():
                            _lbl = (
                                f"[{_r.get('status','?')}]  "
                                f"{_r.get('config_label','?')[:60]}  "
                                f"seed={_r.get('p_seed','?')}  "
                                f"→ {Path(_r['exp_dir']).name}"
                            )
                            _run_labels.append((_lbl, _r["exp_dir"], _r.get("status")))

                        _sel_runs = st.multiselect(
                            "Select runs to delete",
                            options=[l for l, _, _ in _run_labels],
                            key=f"sel_runs_{_sid}",
                            placeholder="Pick one or more runs…",
                        )
                        _confirm_runs_key = f"confirm_del_runs_{_sid}"
                        if _sel_runs:
                            _dirs_to_del = [
                                d for (l, d, _) in _run_labels if l in _sel_runs
                            ]
                            if not st.session_state.get(_confirm_runs_key):
                                if st.button(
                                    f"🗑 Delete {len(_sel_runs)} run(s)",
                                    key=f"del_runs_btn_{_sid}",
                                    type="primary",
                                ):
                                    st.session_state[_confirm_runs_key] = True
                                    st.rerun()
                            else:
                                st.warning(
                                    f"⚠️ This will permanently delete "
                                    f"**{len(_dirs_to_del)} directory/ies** from disk. "
                                    f"This cannot be undone."
                                )
                                _cc1, _cc2 = st.columns(2)
                                if _cc1.button(
                                    "✅ Yes, delete",
                                    key=f"confirm_yes_runs_{_sid}",
                                    type="primary",
                                ):
                                    _errs = []
                                    for _d in _dirs_to_del:
                                        try:
                                            shutil.rmtree(_d)
                                        except OSError as _exc:
                                            _errs.append(f"{_d}: {_exc}")
                                    st.session_state.pop(_confirm_runs_key, None)
                                    st.cache_data.clear()
                                    if _errs:
                                        st.error("Some deletions failed:\n" + "\n".join(_errs))
                                    else:
                                        st.success(f"Deleted {len(_dirs_to_del)} run(s).")
                                    st.rerun()
                                if _cc2.button(
                                    "❌ Cancel",
                                    key=f"confirm_no_runs_{_sid}",
                                ):
                                    st.session_state.pop(_confirm_runs_key, None)
                                    st.rerun()

                    # ── Delete whole sweep ────────────────────────────────────
                    _confirm_sweep_key = f"confirm_del_sweep_{_sid}"
                    if not st.session_state.get(_confirm_sweep_key):
                        if st.button(
                            f"🗑 Delete entire sweep ({row['total']} runs)",
                            key=f"del_sweep_btn_{_sid}",
                        ):
                            st.session_state[_confirm_sweep_key] = True
                            st.rerun()
                    else:
                        st.error(
                            f"⚠️ **Permanently delete all {row['total']} runs** "
                            f"of sweep `{row['sweep_name']}` from disk?  "
                            f"This cannot be undone."
                        )
                        _sc1, _sc2 = st.columns(2)
                        if _sc1.button(
                            "✅ Yes, delete all",
                            key=f"confirm_yes_sweep_{_sid}",
                            type="primary",
                        ):
                            _errs = []
                            for _, _r in sweep_runs.iterrows():
                                try:
                                    shutil.rmtree(_r["exp_dir"])
                                except OSError as _exc:
                                    _errs.append(f"{_r['exp_dir']}: {_exc}")
                            _hidden_sweeps.discard(_sid)
                            _save_hidden_sweeps(_hidden_sweeps)
                            st.session_state.pop(_confirm_sweep_key, None)
                            st.cache_data.clear()
                            if _errs:
                                st.error("Some deletions failed:\n" + "\n".join(_errs))
                            else:
                                st.success(f"Sweep `{row['sweep_name']}` deleted.")
                            st.rerun()
                        if _sc2.button(
                            "❌ Cancel",
                            key=f"confirm_no_sweep_{_sid}",
                        ):
                            st.session_state.pop(_confirm_sweep_key, None)
                            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.header("Configuration Comparison")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch tab.")
    else:
        # Sweep filter
        all_sweep_names = sorted(df["sweep_name"].unique().tolist())
        sel_sweeps = st.multiselect(
            "Filter by sweep(s)  (empty = all sweeps)",
            options=all_sweep_names,
        )
        cmp_df = df[df["sweep_name"].isin(sel_sweeps)] if sel_sweeps else df

        agg_list = get_all_aggregated(cmp_df)
        if not agg_list:
            st.info("No aggregated data available for the selected sweeps.")
        else:
            # Build option labels using short (diff-only) labels so the
            # multiselect is readable — e.g. "tau=0.5" instead of the full string.
            valid_aggs = [a for a in agg_list if a["n_seeds"] > 0]
            if not valid_aggs:
                st.info("All configs have 0 seeds recorded so far.")
            else:
                all_short = _diff_labels(valid_aggs)
                # Deduplicate: if two entries get the same short label, append sweep name
                seen_short: dict[str, int] = {}
                display_keys = []
                for sl, a in zip(all_short, valid_aggs):
                    if all_short.count(sl) > 1:
                        sl = f"{a['sweep_name']} / {sl}"
                    if sl in seen_short:
                        sl = f"{sl} [{a['config_key']}]"
                    seen_short[sl] = 1
                    display_keys.append(sl)

                # Map display key → agg dict
                config_options = dict(zip(display_keys, valid_aggs))

                default_sel = display_keys  # select all by default
                sel_configs = st.multiselect(
                    "Select configurations to compare",
                    options=display_keys,
                    default=default_sel,
                )

                x_axis = st.radio(
                    "X-axis", ["Episode", "Labeled Set Size"], horizontal=True
                )

                if sel_configs:
                    chosen = [config_options[k] for k in sel_configs]
                    short_labels = _diff_labels(chosen)
                    fig = _build_comparison_chart(chosen, x_axis, short_labels=short_labels)
                    st.plotly_chart(fig, width="stretch")

                    # Download chart as standalone HTML
                    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                    st.download_button(
                        "⬇ Download chart as HTML",
                        data=(
                            "<html><head><meta charset='utf-8'></head>"
                            f"<body>{chart_html}</body></html>"
                        ),
                        file_name="al_comparison_chart.html",
                        mime="text/html",
                    )

                    # Summary table
                    st.subheader("Final Accuracy Summary")
                    rows = []
                    for a, sl in zip(chosen, short_labels):
                        rows.append(
                            {
                                "Config": sl,
                                "Seeds": a["n_seeds"],
                                "Final Test Acc (mean)": (
                                    f"{a['final_test_acc_mean']:.2f}%"
                                    if a["final_test_acc_mean"] is not None
                                    else "—"
                                ),
                                "± Std": (
                                    f"{a['final_test_acc_std']:.2f}%"
                                    if a["final_test_acc_std"] is not None
                                    else "—"
                                ),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Select at least one configuration above.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUN DETAILS
# ═════════════════════════════════════════════════════════════════════════════

with tab_details:
    st.header("Run Details")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch tab.")
    else:
        col1, col2, col3 = st.columns(3)
        sweep_names = sorted(df["sweep_name"].unique().tolist())
        sweep_sel = col1.selectbox("Sweep", options=sweep_names)

        sweep_runs = df[df["sweep_name"] == sweep_sel]
        config_names = sorted(sweep_runs["config_label"].unique().tolist())
        config_sel = col2.selectbox("Config", options=config_names)

        config_runs = sweep_runs[sweep_runs["config_label"] == config_sel]
        seed_col = "p_seed" if "p_seed" in config_runs.columns else None
        if seed_col:
            seeds = sorted(config_runs[seed_col].dropna().astype(int).unique().tolist())
            seed_sel = col3.selectbox("Seed", options=seeds if seeds else [0])
            run_row = config_runs[config_runs[seed_col].fillna(-1).astype(int) == seed_sel]
        else:
            seed_sel = None
            run_row = config_runs

        if run_row.empty:
            st.warning("No run found for this selection.")
        else:
            run = run_row.iloc[0]
            status_col, commit_col, dir_col = st.columns([1, 1, 3])
            status_col.metric("Status", _badge(run["status"]))
            commit_col.metric("Commit", str(run["commit_hash"])[:8])
            dir_col.markdown(f"**Exp Dir:** `{run['exp_dir']}`")

            if run.get("error"):
                st.error(f"**Error:** {run['error']}")

            # Parameters
            with st.expander("📋 Run Parameters", expanded=True):
                param_cols = [c for c in run.index if c.startswith("p_")]
                params_dict = {c[2:]: str(run[c]) for c in param_cols if run[c] is not None}
                st.dataframe(
                    pd.DataFrame(
                        list(params_dict.items()), columns=["Parameter", "Value"]
                    ),
                    width="stretch",
                    hide_index=True,
                )

            # Episode table + chart
            episodes = run.get("episodes") or []
            if episodes:
                st.subheader("Per-Episode Results")
                ep_df = pd.DataFrame(episodes)
                ep_rename = {
                    "episode": "Episode",
                    "test_acc": "Test Acc (%)",
                    "val_acc": "Val Acc (%)",
                    "train_acc": "Train Acc (%)",
                    "labeled_size": "Labeled Size",
                    "duration_s": "Duration (s)",
                }
                st.dataframe(
                    ep_df.rename(columns=ep_rename),
                    width="stretch",
                    hide_index=True,
                )

                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(
                        x=ep_df["episode"],
                        y=ep_df["test_acc"],
                        mode="lines+markers",
                        name="Test Acc",
                        line=dict(color=_COLORS[0], width=2),
                    )
                )
                if "val_acc" in ep_df.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=ep_df["episode"],
                            y=ep_df["val_acc"],
                            mode="lines+markers",
                            name="Val Acc",
                            line=dict(color=_COLORS[1], width=2, dash="dash"),
                        )
                    )
                fig2.update_layout(
                    title=f"Accuracy — seed={seed_sel}",
                    xaxis_title="Episode",
                    yaxis_title="Accuracy (%)",
                    template="plotly_white",
                    height=350,
                    margin=dict(t=40),
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("No episode data recorded yet (run may still be starting).")

            # Log viewer
            st.subheader("stdout.log")
            log_path = Path(run["exp_dir"]) / "stdout.log"
            if log_path.exists():
                filter_str = st.text_input(
                    "Filter lines (case-insensitive substring)", key="log_filter"
                )
                try:
                    log_text = log_path.read_text(errors="replace")
                    if filter_str:
                        lines = [
                            ln for ln in log_text.splitlines()
                            if filter_str.lower() in ln.lower()
                        ]
                        log_text = "\n".join(lines)
                    st.text_area(
                        "log", log_text, height=420, label_visibility="collapsed"
                    )
                except OSError as exc:
                    st.warning(f"Could not read log: {exc}")
            else:
                st.info(f"Log not found at `{log_path}`")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ν' CACHE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

with tab_nu:
    st.header("ν' (nu_prime) vs τ — Cache Explorer")
    st.caption(
        "Interactive view of the pre-computed nu_prime cache. "
        "Each curve shows the equilibrium noise tolerance ν' as a function of the cost ratio τ."
    )

    # ── dataset selector ────────────────────────────────────────────────────
    _nu_dataset = st.selectbox(
        "Dataset", options=["cifar100", "cifar10", "imagenet50"], key="nu_dataset"
    )
    _nu_cache = _load_nu_cache(_nu_dataset)

    if not _nu_cache:
        st.warning(
            f"No nu_prime cache found for **{_nu_dataset}**. "
            "Run `python batch_utils/sweep_nu_prime.py --submit` to populate it."
        )
    else:
        _nu_methods_available = sorted(_nu_cache.keys())
        _nu_budgets_available = sorted(
            {int(b) for m in _nu_cache.values() for b in m.keys()}
        )

        # ── filters ─────────────────────────────────────────────────────────
        _fc1, _fc2 = st.columns(2)
        _sel_methods = _fc1.multiselect(
            "AL methods",
            options=_nu_methods_available,
            default=_nu_methods_available,
            key="nu_sel_methods",
        )
        _sel_budgets = _fc2.multiselect(
            "Budgets",
            options=_nu_budgets_available,
            default=_nu_budgets_available,
            key="nu_sel_budgets",
        )
        _show_anchor = st.checkbox(
            "Show oracle anchor (τ=1, ν'=0)", value=True, key="nu_show_anchor"
        )

        if not _sel_methods or not _sel_budgets:
            st.info("Select at least one method and one budget.")
        else:
            _nu_fig = go.Figure()
            _nu_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
            ]
            _nu_dash_styles = ["solid", "dash", "dot", "dashdot"]
            _color_idx = 0

            for _m_idx, _method in enumerate(_sel_methods):
                if _method not in _nu_cache:
                    continue
                for _b_idx, _budget in enumerate(_sel_budgets):
                    _b_key = str(_budget)
                    _tau_dict = _nu_cache[_method].get(_b_key)
                    if not _tau_dict:
                        continue

                    # sort by tau, inject oracle anchor if requested
                    _taus = sorted(float(t) for t in _tau_dict.keys())
                    _nus  = [_tau_dict[f"{t:.4f}"] for t in _taus]
                    if _show_anchor and 1.0 not in _taus:
                        _taus.append(1.0)
                        _nus.append(0.0)

                    _color = _nu_colors[_m_idx % len(_nu_colors)]
                    _dash  = _nu_dash_styles[_b_idx % len(_nu_dash_styles)]
                    _label = f"{_method} / B={_budget}"

                    _nu_fig.add_trace(
                        go.Scatter(
                            x=_taus,
                            y=_nus,
                            mode="lines+markers",
                            name=_label,
                            line=dict(color=_color, width=2.2, dash=_dash),
                            marker=dict(size=7),
                            hovertemplate=(
                                f"<b>{_label}</b><br>"
                                "τ = %{x:.2f}<br>"
                                "ν' = %{y:.4f}<extra></extra>"
                            ),
                        )
                    )

            _nu_fig.add_hline(
                y=0.5, line_dash="dot", line_color="grey",
                annotation_text="ν'=0.5", annotation_position="right",
            )
            _nu_fig.update_layout(
                xaxis_title="τ  (cost ratio = oracle_budget / noisy_size)",
                yaxis_title="ν'  (equilibrium noise tolerance)",
                xaxis=dict(range=[0, 1.05], tickformat=".1f"),
                yaxis=dict(range=[0, 1.0]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                hovermode="x unified",
                template="plotly_white",
                height=500,
                margin=dict(t=40),
                title=f"{_nu_dataset.upper()} — ν' vs τ",
            )
            st.plotly_chart(_nu_fig, use_container_width=True)

            # ── raw data table ───────────────────────────────────────────────
            with st.expander("Raw cache values", expanded=False):
                _rows = []
                for _method in _sel_methods:
                    for _budget in _sel_budgets:
                        _b_key = str(_budget)
                        _td = _nu_cache.get(_method, {}).get(_b_key, {})
                        for _t_key, _nu_val in sorted(_td.items(), key=lambda x: float(x[0])):
                            _rows.append({
                                "method": _method,
                                "budget": _budget,
                                "tau": float(_t_key),
                                "nu_prime": _nu_val,
                            })
                        if _show_anchor:
                            _rows.append({
                                "method": _method,
                                "budget": _budget,
                                "tau": 1.0,
                                "nu_prime": 0.0,
                            })
                if _rows:
                    st.dataframe(
                        pd.DataFrame(_rows).sort_values(["method", "budget", "tau"]),
                        hide_index=True,
                        use_container_width=True,
                    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT  (was TAB 4)
# ═════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("Export Report")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch tab.")
    else:
        st.markdown(
            "Generate a **fully self-contained HTML file** with comparison charts "
            "and a summary table.  The file can be opened offline and shared via "
            "email or cloud storage — no VPN or server connection required."
        )

        exp_sweeps = st.multiselect(
            "Sweeps to include  (empty = all)",
            options=sorted(df["sweep_name"].unique().tolist()),
            key="exp_sweeps",
        )
        export_df = df[df["sweep_name"].isin(exp_sweeps)] if exp_sweeps else df

        exp_configs = st.multiselect(
            "Configs to include  (empty = all)",
            options=sorted(export_df["config_label"].unique().tolist()),
            key="exp_configs",
        )
        if exp_configs:
            export_df = export_df[export_df["config_label"].isin(exp_configs)]

        report_title = st.text_input("Report title", value="AL Experiment Report")

        col_info, col_btn = st.columns([3, 1])
        col_info.markdown(f"**{len(export_df)} run(s)** will be included.")

        if col_btn.button("Generate", width="stretch"):
            with st.spinner("Building HTML report…"):
                html_content = generate_report(export_df, title=report_title)
            st.download_button(
                "⬇ Download report.html",
                data=html_content,
                file_name="al_report.html",
                mime="text/html",
                width="stretch",
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — LAUNCH
# ═════════════════════════════════════════════════════════════════════════════

with tab_launch:
    st.header("🚀 Launch Sweep")

    with st.expander("ℹ️ Where to run — setup guide", expanded=False):
        st.markdown(
            """
**Run the dashboard on: `takara`** (or any GPU node with the venv).  
The SLURM cluster is accessed via `phoenix-gw`, but you never need to be *on* phoenix-gw manually.

| Button | What it does | Where to run the dashboard |
|---|---|---|
| ▶ Run locally | Runs all jobs sequentially on **this machine** (takara) | takara |
| 📤 Submit to SLURM | SSHes to `phoenix-gw` and calls `sbatch` automatically | takara |

**Quick start on takara:**
```bash
cd /cs/labs/daphna/alongol/projects/daphna_lab
source venv/bin/activate
streamlit run reporting/dashboard.py
```
Then on your laptop open the SSH tunnel:
```bash
ssh -L 8501:localhost:8501 alongol@takara
```
And browse to **http://localhost:8501**.
            """
        )

    # ── SSH key check for SLURM submission ───────────────────────────────────
    _slurm_node = getattr(swt if 'swt' in dir() else type('_', (), {})(), 'SLURM_LOGIN_NODE', 'phoenix-gw') \
        if False else 'phoenix-gw'  # read after swt is imported below

    def _check_ssh_auth(host: str) -> tuple[bool, str]:
        """Return (ok, message) for passwordless SSH to host."""
        if not shutil.which("ssh"):
            return False, "'ssh' not found on this machine."
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             host, "echo ok"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return True, f"Passwordless SSH to `{host}` ✅"
        return False, r.stderr.strip() or f"SSH to `{host}` requires a password or key."

    def _setup_ssh_key(host: str, password: str) -> tuple[bool, str]:
        """Generate an ed25519 key (if missing), then copy it to host via paramiko."""
        import paramiko
        key_path = Path.home() / ".ssh" / "id_ed25519"
        pub_path = Path(str(key_path) + ".pub")

        # Generate key pair if missing
        if not key_path.exists():
            key = paramiko.Ed25519Key.generate()
            key_path.parent.mkdir(mode=0o700, exist_ok=True)
            key_path.touch(mode=0o600)
            key_path.write_text(key.get_base64())  # private key via paramiko
            # Use ssh-keygen for proper PEM format instead
            r = subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                return False, f"ssh-keygen failed: {r.stderr.strip()}"

        if not pub_path.exists():
            return False, f"Public key not found at `{pub_path}`."

        pub_key_line = pub_path.read_text().strip()

        # Connect with password and append public key to authorized_keys
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=os.environ.get("USER", ""), password=password,
                           timeout=10, allow_agent=False, look_for_keys=False)
            sftp = client.open_sftp()
            try:
                sftp.stat(".ssh")
            except FileNotFoundError:
                sftp.mkdir(".ssh")
                sftp.chmod(".ssh", 0o700)
            ak_path = ".ssh/authorized_keys"
            existing = ""
            try:
                with sftp.open(ak_path, "r") as f:
                    existing = f.read().decode()
            except FileNotFoundError:
                pass
            if pub_key_line not in existing:
                with sftp.open(ak_path, "a") as f:
                    f.write(f"\n{pub_key_line}\n")
            sftp.chmod(ak_path, 0o600)  # always enforce correct permissions
            sftp.close()
            client.close()
            return True, f"Key copied to `{host}`. Passwordless SSH is now set up ✅"
        except paramiko.AuthenticationException:
            return False, "Wrong password — please try again."
        except Exception as exc:
            return False, f"Connection to `{host}` failed: {exc}"

    # ── Load defaults from sweep_train_al.py ─────────────────────────────────
    try:
        swt = _import_swt()
        _slurm_node = getattr(swt, 'SLURM_LOGIN_NODE', 'phoenix-gw')
        _default_al = swt.AL_CONFIGS
        _default_grid = swt.GRID
        _default_fixed = swt.FIXED_ARGS.strip()
    except Exception as _e:
        st.error(f"Could not import sweep_train_al.py: {_e}")
        swt = None
        _default_al = ["--al probcover"]
        _default_grid = {"seed": [1, 2, 3]}
        _default_fixed = ""

    # ── Config editor ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        al_text = st.text_area(
            "AL\_CONFIGS — one variant per line",
            value="\n".join(_default_al),
            height=170,
            help="Each line is one AL experiment variant. Lines are NOT crossed with each other.",
            key="launch_al_configs",
        )
        fixed_args_input = st.text_input(
            "FIXED\_ARGS (added to every job)",
            value=_default_fixed,
            key="launch_fixed_args",
        )

    with col_right:
        _grid_default_text = "\n".join(
            f"{k}={','.join(str(v) for v in vals)}"
            for k, vals in _default_grid.items()
        )
        grid_text = st.text_area(
            "GRID — key=val1,val2,... per line (fully crossed with each AL\_CONFIG)",
            value=_grid_default_text,
            height=170,
            help="e.g.\nseed=1,2,3\ninitial_delta=0.75,1",
            key="launch_grid",
        )

    # ── Parse ─────────────────────────────────────────────────────────────────
    al_configs_parsed = [l.strip() for l in al_text.splitlines() if l.strip()]
    grid_parsed: dict = {}
    for line in grid_text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, vals_str = line.split("=", 1)
        parsed_vals = []
        for v in vals_str.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                parsed_vals.append(float(v) if "." in v else int(v))
            except ValueError:
                parsed_vals.append(v)
        grid_parsed[k.strip()] = parsed_vals

    parsed_ok = bool(al_configs_parsed and grid_parsed and swt)
    if parsed_ok:
        try:
            preview_jobs = swt.build_jobs_from(
                al_configs_parsed, grid_parsed, fixed_args_input
            )
        except Exception as _pe:
            st.error(f"Job generation error: {_pe}")
            preview_jobs = []
            parsed_ok = False
    else:
        preview_jobs = []

    n_combos = len(list(itertools.product(*grid_parsed.values()))) if grid_parsed else 0
    st.caption(
        f"**{len(preview_jobs)} jobs** "
        f"({len(al_configs_parsed)} AL config(s) × {n_combos} grid combo(s))"
    )

    with st.expander("Preview job commands", expanded=False):
        if preview_jobs:
            for idx, j in enumerate(preview_jobs):
                st.code(f"[{idx}] {j}", language=None)
        else:
            st.info("Fix the config above to see a preview.")

    st.divider()

    # ── Launch buttons ────────────────────────────────────────────────────────
    col_local, col_sbatch = st.columns(2)

    with col_local:
        st.markdown("**▶ Run locally** — on this machine (takara)")
        st.caption("Jobs run sequentially in a detached background process. Safe to close this tab.")
        if st.button(
            "▶ Run locally",
            disabled=not parsed_ok or not preview_jobs,
            width="stretch",
            key="btn_local",
        ):
            sweep_name = swt.build_sweep_name_from(al_configs_parsed, grid_parsed)
            with st.spinner(f"Launching {len(preview_jobs)} jobs…"):
                info = swt.launch_local_bg(preview_jobs, sweep_name)
            launches = _load_launches()
            launches.append(info)
            _save_launches(launches)
            st.session_state["last_launched_sweep_id"] = info["sweep_id"]
            st.success(
                f"Started! PID={info['pid']} | sweep_id=`{info['sweep_id']}`  \n"
                f"Log: `{info['log_file']}`"
            )
            st.info("📊 Switch to the **Sweeps** tab — your new sweep is highlighted there.")
            st.cache_data.clear()  # so the Sweeps tab picks up the new sweep

    with col_sbatch:
        st.markdown(f"**📤 Submit to SLURM** — via `{_slurm_node}`")
        # Show SSH connectivity status
        _ssh_ok, _ssh_msg = _check_ssh_auth(_slurm_node)
        if _ssh_ok:
            st.caption(_ssh_msg)
        else:
            st.warning(f"No passwordless SSH to `{_slurm_node}` yet — SLURM submission will fail.")
            _ssh_password = st.text_input(
                f"Your password for `{_slurm_node}`",
                type="password",
                key="ssh_password_input",
                help="Used once to copy your SSH key. Never stored anywhere.",
            )
            if st.button("🔑 Set up passwordless SSH now", key="btn_setup_ssh", width="stretch",
                         disabled=not _ssh_password):
                with st.spinner(f"Copying SSH key to {_slurm_node}…"):
                    _setup_ok, _setup_msg = _setup_ssh_key(_slurm_node, _ssh_password)
                if _setup_ok:
                    st.success(_setup_msg)
                else:
                    st.error(_setup_msg)
        if st.button(
            "📤 Submit to SLURM",
            disabled=not parsed_ok or not preview_jobs,
            width="stretch",
            key="btn_sbatch",
        ):
            with st.spinner(f"SSHing to {_slurm_node} and submitting…"):
                result = swt.submit_sbatch_jobs(preview_jobs)
            if result["returncode"] == 0:
                st.success(f"Submitted: {result['stdout']}")
            else:
                st.error(f"sbatch failed: {result['stderr']}")

    st.divider()

    # ── Active / recent launches ──────────────────────────────────────────────
    _lhdr_col, _lclear_col = st.columns([5, 1])
    _lhdr_col.subheader("Recent Launches")
    _launches = _load_launches()
    if _launches and _lclear_col.button(
        "🗑 Clear done",
        key="clear_done_launches",
        help="Remove all non-running launches from this list",
    ):
        _launches = [l for l in _launches if _is_process_running(l.get("pid"))]
        _save_launches(_launches)
        st.rerun()

    if not _launches:
        st.info("No launches yet. Use the buttons above to start your first sweep.")
    else:
        _last_sweep_id_ln = st.session_state.get("last_launched_sweep_id")
        for _launch in reversed(_launches):
            _pid = _launch.get("pid")
            _mode = _launch.get("mode", "?")
            _running = _is_process_running(_pid)
            _is_latest = _launch.get("sweep_id") == _last_sweep_id_ln
            _icon = "🔄" if _running else "✅"
            _header = (
                f"{_icon} {_launch.get('sweep_name', '?')}  "
                f"| {_launch.get('started_at', '')[:19]}  "
                f"| {_launch.get('total_jobs', '?')} jobs  "
                f"| {_mode}"
            )
            with st.expander(_header, expanded=_running or _is_latest):
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Mode", _mode)
                _c2.metric("Total jobs", _launch.get("total_jobs", "?"))
                _c3.markdown(f"**PID:** `{_pid}`")
                st.markdown(f"**Sweep ID:** `{_launch.get('sweep_id', '?')}`")

                _log_path = Path(_launch.get("log_file", ""))
                if _log_path.exists():
                    _tail_n = st.number_input(
                        "Lines to show",
                        min_value=20, max_value=1000, value=60,
                        key=f"tail_{_launch.get('sweep_id', id(_launch))}",
                    )
                    _log_lines = _log_path.read_text(errors="replace").splitlines()
                    st.text_area(
                        "log",
                        value="\n".join(_log_lines[-int(_tail_n):]),
                        height=380,
                        key=f"log_{_launch.get('sweep_id', id(_launch))}",
                        label_visibility="collapsed",
                    )
                    st.caption(f"`{_log_path}`")
                    if st.button(
                        "🔄 Refresh log",
                        key=f"refresh_{_launch.get('sweep_id', id(_launch))}",
                    ):
                        st.rerun()
                elif _mode == "local":
                    st.info("Log file not yet available.")

                st.divider()
                if st.button(
                    "🗑 Remove from list",
                    key=f"del_{_launch.get('sweep_id', id(_launch))}",
                    help="Remove this entry from launch history (does not delete any output files)",
                ):
                    _updated = [
                        l for l in _load_launches()
                        if l.get("sweep_id") != _launch.get("sweep_id")
                    ]
                    _save_launches(_updated)
                    st.rerun()

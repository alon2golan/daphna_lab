"""
AL Dashboard — Streamlit experiment dashboard for the active-learning sweeps.

    streamlit run reporting/dashboard.py --server.port 8501

Three top-level tabs, each with sub-tabs, grouped by data source:
  1. 📋 Sweeps        – 📒 Ledger:  general SLURM job ledger (ALL kinds), from the
                          registry (logs/sweep/sweep_registry.json). Grouped by
                          category; per-sweep override. Seed via
                          `python reporting/backfill_registry.py`.
                        🚀 Launch:  configure + fire new train_al sweeps (local or
                          SLURM via phoenix-gw) and tail recent launches.
  2. 📈 Graphs         – 📈 Comparison: train_al comparison (result.json) with the
                          full filter set (date, al_method, τ, ν, B, …).
                        🧪 NPC Repro:   Netta's noise-aware ProbCover panels
                          (NPC vs ProbCover vs Random, one panel per noise level).
                        🔍 Run Details: per-run params, episodes, and log viewer.
                        📤 Export:      self-contained HTML report download.
  3. 📊 Additional    – ν' cache explorer + a generic artifact browser over
                        results/** (renders .png / .csv / .json).

It reads shared modules (reporting/scanner.py, registry.py, html_export.py) and
the on-disk launch/registry state under logs/.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reporting.scanner import (  # noqa: E402
    _LABEL_KEY_ORDER,
    describe_sweep,
    get_all_aggregated,
    load_all_results,
)
from reporting import registry as reg  # noqa: E402
from reporting.html_export import generate_report  # noqa: E402
from deep_al.pycls.noise.nu_prime_cache import load_cache as _load_nu_cache  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AL Dashboard (next)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_RESULTS_DIR = REPO_ROOT / "results"

_STATUS_EMOJI = {"completed": "✅", "running": "🔄", "failed": "❌"}

# Persisted launches log (survives page refresh).
_LAUNCHES_FILE = REPO_ROOT / "logs" / "sweep" / "dashboard_launches.json"


def _badge(status: str) -> str:
    return f"{_STATUS_EMOJI.get(status, '❓')} {status}"


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


with st.sidebar:
    st.title("🧪 AL Dashboard")
    output_dir = st.text_input("Output directory", value=str(REPO_ROOT / "output"))
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Auto-refreshes every 30 s.")


# ── Data loading ────────────────────────────────────────────────────────────────


@st.cache_data(ttl=30, show_spinner="Scanning result.json…")
def _load_results(directory: str) -> pd.DataFrame:
    return load_all_results(directory)


def _registry_category_map() -> dict[str, str]:
    return {e["sweep_id"]: e.get("category", "uncategorized") for e in reg.load_registry()}


def _registry_entry_map() -> dict[str, dict]:
    """sweep_id → full registry entry (title/context/flags/note/category)."""
    return {e["sweep_id"]: e for e in reg.load_registry() if e.get("sweep_id")}


# ── Shareable URL state (Feature 4) ────────────────────────────────────────────
#
# The Graphs comparison view is encoded in st.query_params so a link restores
# filters + grouping + axes + selection. We only ever READ query params to seed a
# widget's initial index/default (taking effect on first render only — Streamlit
# ignores index/default once the widget owns a session_state value), and WRITE
# back only changed keys. We never write session_state for a keyed widget, so
# there are no "default + session_state" warnings and no rerun loops.


def _qp_get(key: str, default: str = "") -> str:
    v = st.query_params.get(key)
    return v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else default)


def _qp_list(key: str) -> list[str]:
    raw = _qp_get(key)
    return [x for x in raw.split(",") if x] if raw else []


def _qp_index(options: list, value, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


def _qp_write(pairs: dict):
    """Set changed keys; delete keys whose value is None/''. Idempotent → stable."""
    for k, v in pairs.items():
        if v is None or v == "":
            if k in st.query_params:
                del st.query_params[k]
        elif st.query_params.get(k) != v:
            st.query_params[k] = v


df = _load_results(output_dir)


# ── Shared chart helpers ──


def _diff_labels(agg_list: list) -> list[str]:
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
    diff_keys = [k for k in all_keys if len({p.get(k) for p in parsed}) > 1]
    if not diff_keys:
        return [a["sweep_name"] for a in agg_list]
    ordered = [k for k in _LABEL_KEY_ORDER if k in diff_keys]
    ordered += sorted(k for k in diff_keys if k not in ordered)
    short = []
    for p in parsed:
        parts = [f"{k}={p[k] if k in p else '—'}" for k in ordered]
        short.append(" | ".join(parts) if parts else "?")
    return short


def _sweep_tag(sweep_name: str) -> str:
    """Compact sweep identifier for legends: sweep_name minus its timestamp."""
    import re
    return re.sub(r"_\d{8}_\d{6}$", "", sweep_name or "") or (sweep_name or "?")


def _labels_with_sweep(agg_list: list) -> list[str]:
    """
    Like _diff_labels, but when the curves span MORE THAN ONE sweep, prefix each
    label with a short sweep tag so you can tell which curve came from which
    sweep. Within a single sweep, returns the clean param-only diff labels.
    """
    param = _diff_labels(agg_list)
    if len({a["sweep_name"] for a in agg_list}) <= 1:
        return param
    out = []
    for lab, a in zip(param, agg_list):
        tag = _sweep_tag(a["sweep_name"])
        # When _diff_labels fell back to the sweep_name (no param differs), the
        # tag alone is enough; otherwise show "tag · param=…".
        out.append(tag if (not lab or lab == a["sweep_name"]) else f"{tag} · {lab}")
    return out


# X-axis options (Feature 2): display label → aggregated-dataframe column.
_X_AXES = {
    "Episode": "episode",
    "# labeled examples": "labeled_size",
    "Oracle budget units": "budget_units",
}
_X_TITLE = {
    "episode": "Episode",
    "labeled_size": "# labeled examples",
    "budget_units": "Oracle budget units (= #examples × τ)",
}
# Y-axis: known per-episode metrics shown first; the rest discovered dynamically.
_Y_ORDER = ["test_acc", "val_acc", "train_acc", "duration_s"]
_Y_TITLE = {
    "test_acc": "Test Accuracy (%)", "val_acc": "Validation Accuracy (%)",
    "train_acc": "Train Accuracy (%)", "duration_s": "Episode duration (s)",
}


def _y_axis_options(agg_list) -> list[str]:
    """Per-episode numeric metrics available across the selected curves."""
    cols: set[str] = set()
    for a in agg_list:
        if not a["agg"].empty:
            cols |= {c[:-5] for c in a["agg"].columns if c.endswith("_mean")}
    ordered = [c for c in _Y_ORDER if c in cols]
    ordered += sorted(c for c in cols if c not in ordered)
    return ordered or ["test_acc"]


def _build_comparison_chart(agg_list, x_field="episode", y_field="test_acc",
                            short_labels=None, styles=None, y_range=None,
                            title=None, color_offset=0) -> go.Figure:
    fig = go.Figure()
    y_mean, y_std = f"{y_field}_mean", f"{y_field}_std"
    is_pct = y_field.endswith("_acc")
    for i, agg in enumerate(agg_list):
        a = agg["agg"]
        if a.empty or x_field not in a.columns or y_mean not in a.columns:
            continue
        a = a.reset_index(drop=True)
        color = _COLORS[(i + color_offset) % len(_COLORS)]
        flagged = bool(styles[i].get("flagged")) if styles else False
        x = list(a[x_field])
        mean = list(a[y_mean])
        std = list(a[y_std]) if y_std in a.columns else [0.0] * len(mean)
        label = (short_labels[i] if short_labels else None) or \
            f"{agg['sweep_name']} / {agg['config_label']}"
        if flagged:
            label = f"⚠ {label}"
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1], fill="toself",
            fillcolor=("#999999" if flagged else color),
            opacity=0.10 if flagged else 0.15, line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines+markers", name=f"{label} (n={agg['n_seeds']})",
            line=dict(color=("#999999" if flagged else color), width=2.5,
                      dash="dash" if flagged else None),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{label}</b><br>{_X_TITLE.get(x_field, x_field)}: %{{x}}<br>"
                + (f"{_Y_TITLE.get(y_field, y_field)}: %{{y:.2f}}"
                   + ("%" if is_pct else ""))
                + "<extra></extra>"
            )))
    fig.update_layout(
        xaxis_title=_X_TITLE.get(x_field, x_field),
        yaxis_title=_Y_TITLE.get(y_field, y_field),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", template="plotly_white", height=480,
        margin=dict(t=40), title=title)
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


def _shared_y_range(panel_agg_lists, y_field):
    """Common y-axis range across facet panels so they're visually comparable."""
    ym, ys = f"{y_field}_mean", f"{y_field}_std"
    lo = hi = None
    for aggs in panel_agg_lists:
        for a in aggs:
            d = a["agg"]
            if d.empty or ym not in d.columns:
                continue
            s = d[ys] if ys in d.columns else 0.0
            clo, chi = float((d[ym] - s).min()), float((d[ym] + s).max())
            lo = clo if lo is None else min(lo, clo)
            hi = chi if hi is None else max(hi, chi)
    if lo is None:
        return None
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


# ── NPC reproduction helpers (Netta's noise-aware ProbCover) ───────────────────

# Fixed method → colour map so a method keeps its colour across every noise panel.
_NPC_METHOD_ORDER = ["npc", "probcover", "random"]
_NPC_METHOD_COLOR = {"npc": "#2ca02c", "probcover": "#1f77b4", "random": "#7f7f7f"}
_NPC_METHOD_LABEL = {"npc": "NPC", "probcover": "ProbCover", "random": "Random"}


def _npc_noise_of(label: str):
    import re
    m = re.search(r"NETTA_n(\d+)", label or "")
    return int(m.group(1)) if m else None


def _npc_method_of(label: str):
    import re
    m = re.search(r"al_method=(\w+)", label or "")
    return m.group(1) if m else None


def _build_npc_panel(agg_by_method: dict, x_axis: str) -> go.Figure:
    """One noise-level panel: a curve per method, fixed colour, ±std band."""
    fig = go.Figure()
    x_col = "labeled_size" if x_axis == "Labeled Set Size" else "episode"
    for method in _NPC_METHOD_ORDER:
        agg = agg_by_method.get(method)
        if not agg or agg["agg"].empty:
            continue
        color = _NPC_METHOD_COLOR[method]
        a = agg["agg"].reset_index(drop=True)
        x = list(a[x_col])
        mean = list(a["test_acc_mean"])
        std = list(a["test_acc_std"])
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1], fill="toself",
            fillcolor=color, opacity=0.13, line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        label = f"{_NPC_METHOD_LABEL[method]} (n={agg['n_seeds']})"
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines+markers", name=label,
            line=dict(color=color, width=2.5), marker=dict(size=6),
            hovertemplate=(
                f"<b>{_NPC_METHOD_LABEL[method]}</b><br>"
                + ("Episode: %{x}" if x_axis == "Episode" else "Labeled: %{x}")
                + "<br>Test Acc: %{y:.2f}%<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title=x_axis, yaxis_title="Test Accuracy (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified", template="plotly_white",
        height=420, margin=dict(t=40, b=40),
    )
    return fig


# ── Tab content renderers (composed into the 3-tab layout at the bottom) ───────

# ── Description cards & auto-facts (Feature 0) + Reproduce panel (Feature 5a) ───


def _flag_badge(flags: list) -> str:
    """A compact ⚠ marker for a flagged sweep, e.g. '⚠ deprecated, bug'."""
    flags = [f for f in (flags or []) if f]
    return f"⚠ {', '.join(flags)}" if flags else ""


def _suggest_title(entry: dict, facts: dict) -> str:
    """Auto-suggested starter title for an un-annotated sweep."""
    name = entry.get("sweep_name", "") or entry.get("sweep_id", "")
    if facts and facts.get("varying"):
        return f"{name} — varies {', '.join(facts['varying'].keys())}"
    return name or "(unnamed sweep)"


def _fmt_date_range(facts: dict) -> str:
    dmin, dmax = facts.get("date_min"), facts.get("date_max")
    if dmin is None:
        return "—"
    s = pd.Timestamp(dmin).strftime("%Y-%m-%d %H:%M")
    if dmax is not None and pd.Timestamp(dmax).date() != pd.Timestamp(dmin).date():
        s += " → " + pd.Timestamp(dmax).strftime("%Y-%m-%d %H:%M")
    return s


def _render_auto_facts(facts: dict):
    """Read-only live auto-fact block (never stored): date, runs, params, command."""
    if not facts:
        st.caption("No live run data (not a train_al run, or no result.json yet).")
        return
    c1, c2 = st.columns(2)
    c1.markdown(f"**📅 Date:** {_fmt_date_range(facts)}")
    sc = facts.get("status_counts") or {}
    sc_str = "  ".join(f"{_STATUS_EMOJI.get(k, '❓')}{v}" for k, v in sc.items()) or "—"
    c2.markdown(f"**🧮 Runs:** {facts.get('n_runs', 0)}  ·  {sc_str}")

    varying = facts.get("varying") or {}
    if varying:
        st.markdown("**🔀 Varying params**")
        st.dataframe(
            pd.DataFrame(
                [{"param": k, "values": ", ".join(str(x) for x in v)}
                 for k, v in varying.items()]
            ),
            width="stretch", hide_index=True,
        )
    constant = facts.get("constant") or {}
    if constant:
        st.caption("**Fixed:** " + "  ·  ".join(f"{k}={v}" for k, v in constant.items()))
    if facts.get("command"):
        st.code(facts["command"], language="bash")
    meta = []
    if facts.get("commit_hash"):
        meta.append(f"commit `{str(facts['commit_hash'])[:8]}`")
    if facts.get("exp_dir"):
        meta.append(f"dir `{facts['exp_dir']}`")
    if meta:
        st.caption("  ·  ".join(meta))


def _render_reproduce(facts: dict, key_prefix: str):
    """Feature 5a: the exact command + code state to reproduce a curve."""
    if not facts or not facts.get("command"):
        return
    with st.expander("🔁 Reproduce this curve", expanded=False):
        st.caption("Exact command + code state used — copy-paste ready.")
        st.code(facts["command"], language="bash")
        st.markdown(
            f"**commit:** `{str(facts.get('commit_hash', '?'))[:12]}` &nbsp;|&nbsp; "
            f"**exp_dir:** `{facts.get('exp_dir', '?')}`"
        )
        st.caption(
            "Full working-tree code snapshot + `git diff vs base` + isolated "
            "worktree command land with Feature 5b (dependency vault)."
        )


def _render_sweep_card(sweep_id, sub_df, *, editable: bool, key_prefix: str,
                       show_reproduce: bool = False):
    """
    Shared description card (Feature 0): curated title/context (stored) + live
    auto-facts (computed). Editable in the Ledger, read-only in Graphs.
    """
    entry = _registry_entry_map().get(str(sweep_id), {})
    facts = describe_sweep(sub_df) if (sub_df is not None and not sub_df.empty) else {}
    title = entry.get("title", "") or ""
    context = entry.get("context", "") or ""

    if editable:
        nt = st.text_input(
            "Title", value=title, key=f"{key_prefix}_title",
            placeholder=_suggest_title(entry, facts),
        )
        nc = st.text_area(
            "Context — the *why* (markdown)", value=context, height=100,
            key=f"{key_prefix}_ctx",
            placeholder="e.g. fixed the ν' cache and re-ran the noisy_vs_oracle sweep",
        )
        if st.button("💾 Save description", key=f"{key_prefix}_savedesc"):
            reg.set_description(str(sweep_id), title=nt, context=nc)
            st.toast("Description saved.")
            st.rerun()
    else:
        if title:
            st.markdown(f"#### {title}")
        if context:
            st.markdown(context)
        if not title and not context:
            st.caption(f"_{_suggest_title(entry, facts)}_  ·  no curated description yet")

    _render_auto_facts(facts)
    if show_reproduce:
        _render_reproduce(facts, key_prefix)


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEPS LEDGER (registry-backed; all job kinds)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_ledger():
    st.header("Sweep ledger")
    st.caption(
        "Every job you've run — train_al, ν'-caching, other repos, misc — from "
        "`logs/sweep/sweep_registry.json`. Re-seed with "
        "`python reporting/backfill_registry.py`."
    )

    entries = reg.load_registry()
    if not entries:
        st.warning(
            "Registry is empty. Run `python reporting/backfill_registry.py` to "
            "seed it from your history."
        )
    else:
        # Live train_al progress per sweep_id from result.json.
        prog: dict[str, dict] = {}
        if not df.empty:
            for sid, g in df.groupby("sweep_id"):
                vc = g["status"].value_counts().to_dict()
                prog[str(sid)] = {
                    "total": len(g),
                    "completed": vc.get("completed", 0),
                    "running": vc.get("running", 0),
                    "failed": vc.get("failed", 0),
                }

        reg_df = pd.DataFrame(entries)
        if "flags" not in reg_df.columns:
            reg_df["flags"] = [[] for _ in range(len(reg_df))]
        reg_df["flags"] = reg_df["flags"].apply(lambda f: f if isinstance(f, list) else [])
        all_cats = sorted(reg_df["category"].unique())
        all_kinds = sorted(reg_df["kind"].unique())

        fc1, fc2, fc3, fc4 = st.columns([2, 2, 1.2, 1])
        sel_cats = fc1.multiselect("Category (empty = all)", options=all_cats)
        sel_kinds = fc2.multiselect("Kind (empty = all)", options=all_kinds)
        show_hidden = fc3.checkbox(
            "Show deprecated/buggy", value=False,
            help="Reveal sweeps flagged ⚠ deprecated or bug (hidden by default).",
        )
        if fc4.button("🔁 Rescan SLURM", help="Re-run backfill incl. one-shot sacct"):
            with st.spinner("Running backfill (ssh sacct)…"):
                r = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "reporting" / "backfill_registry.py")],
                    capture_output=True, text=True, timeout=120,
                )
            st.code((r.stdout or "") + (r.stderr or ""))
            st.rerun()

        view = reg_df
        if sel_cats:
            view = view[view["category"].isin(sel_cats)]
        if sel_kinds:
            view = view[view["kind"].isin(sel_kinds)]
        n_hidden = 0
        if not show_hidden:
            is_hidden = view["flags"].apply(lambda f: bool(set(f) & reg.HIDDEN_FLAGS))
            n_hidden = int(is_hidden.sum())
            view = view[~is_hidden]

        hidden_note = f"  ·  {n_hidden} hidden (⚠ flagged)" if n_hidden else ""
        st.write(f"**{len(view)}** sweeps{hidden_note}")

        # Group by category, known order first.
        ordered_cats = [c for c in reg.KNOWN_CATEGORIES if c in view["category"].unique()]
        ordered_cats += [c for c in sorted(view["category"].unique()) if c not in ordered_cats]

        for cat in ordered_cats:
            cat_rows = view[view["category"] == cat]
            st.subheader(f"{cat}  ·  {len(cat_rows)}")
            for ridx, (_, e) in enumerate(cat_rows.iterrows()):
                sid = e["sweep_id"]
                flags = e.get("flags") if isinstance(e.get("flags"), list) else []
                p = prog.get(sid)
                if p:
                    badge = f"✅{p['completed']} 🔄{p['running']} ❌{p['failed']} / {p['total']}"
                else:
                    badge = f"_{e['kind']}_"
                date = (e.get("created_at") or "")[:16].replace("T", " ")
                ukey = f"{sid}_{ridx}"
                fbadge = _flag_badge(flags)
                title_line = f"**{e['sweep_name']}** — {badge}  ·  {date}"
                if fbadge:
                    title_line = f"{fbadge}  ·  " + title_line
                with st.expander(title_line):
                    st.markdown(
                        f"**id:** `{sid}` &nbsp;|&nbsp; **kind:** `{e['kind']}` "
                        f"&nbsp;|&nbsp; **source:** `{e.get('source','?')}`"
                    )
                    if p:
                        st.progress(
                            p["completed"] / p["total"] if p["total"] else 0.0,
                            text=f"{p['completed']}/{p['total']} completed",
                        )

                    # ── Description card (Feature 0): editable title/context + facts
                    sub_df = df[df["sweep_id"].astype(str) == str(sid)] if not df.empty else df
                    _render_sweep_card(sid, sub_df, editable=True, key_prefix=f"led_{ukey}")

                    new_cat = st.selectbox(
                        "Category", options=reg.KNOWN_CATEGORIES,
                        index=reg.KNOWN_CATEGORIES.index(cat)
                        if cat in reg.KNOWN_CATEGORIES else 0,
                        key=f"cat_{ukey}",
                    )
                    if new_cat != cat:
                        reg.set_category(sid, new_cat)
                        st.rerun()

                    # ── Flags (Feature 1): mark bug / deprecate / restore + note ──
                    st.markdown("**Status flags**")
                    note_val = st.text_input(
                        "Reason / note", value=e.get("note", "") or "",
                        key=f"note_{ukey}", placeholder="why is this flagged? (optional)",
                    )
                    bcol1, bcol2, bcol3 = st.columns(3)
                    if bcol1.button("🐞 Mark bug", key=f"bug_{ukey}",
                                    disabled="bug" in flags):
                        reg.set_flags(sid, list(dict.fromkeys(flags + ["bug"])), note=note_val)
                        st.rerun()
                    if bcol2.button("🗑 Deprecate", key=f"dep_{ukey}",
                                    disabled="deprecated" in flags):
                        reg.set_flags(sid, list(dict.fromkeys(flags + ["deprecated"])), note=note_val)
                        st.rerun()
                    if bcol3.button("↩ Restore", key=f"res_{ukey}", disabled=not flags):
                        reg.set_flags(sid, [], note=note_val)
                        st.rerun()
                    if note_val != (e.get("note", "") or ""):
                        if st.button("💾 Save note", key=f"savenote_{ukey}"):
                            reg.set_flags(sid, flags, note=note_val)
                            st.rerun()
                    if flags:
                        st.caption(f"⚠ flagged: {', '.join(flags)}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHS › COMPARISON (train_al only; full filters)
# ═══════════════════════════════════════════════════════════════════════════════

# Group-by options (Feature 3): friendly name → run-dataframe column.
_GROUPABLE = [
    ("sweep", "sweep_name"),
    ("al_method", "p_al_method"), ("tau", "p_tau"), ("target_nu", "p_target_nu"),
    ("nu_prime", "p_nu_prime"), ("budget", "p_budget"), ("max_iter", "p_max_iter"),
    ("dataset", "p_dataset"), ("model", "p_model"),
    ("eval_model_type", "p_eval_model_type"), ("features_tag", "p_features_tag"),
    ("category", "category"),
]


def _groupable_options(scoped, chosen) -> list[str]:
    """
    Friendly group keys to split panels by. Param keys are offered only when they
    actually vary across the chosen curves; "sweep" is ALWAYS offered (a primary
    organizing dimension) as long as there's run data.
    """
    chosen_keys = {a["config_key"] for a in chosen}
    sub = scoped[scoped["config_key"].isin(chosen_keys)]
    opts = []
    for name, col in _GROUPABLE:
        if col not in sub.columns:
            continue
        if name == "sweep" and not sub.empty:
            opts.append(name)
        elif name != "sweep" and sub[col].nunique(dropna=True) > 1:
            opts.append(name)
    return opts


def _cfgkey_value_map(scoped, col) -> dict:
    """config_key → that config's value of *col* (constant within a config_key)."""
    out = {}
    for ck, g in scoped.groupby("config_key"):
        vals = g[col].dropna().unique()
        out[ck] = vals[0] if len(vals) else None
    return out


def _render_facets(scoped, chosen, labels, styles, x_field, y_field, group_label):
    """Feature 3: one panel per distinct value of the group key, shared y-range."""
    col = dict(_GROUPABLE)[group_label]
    by_sweep = col == "sweep_name"  # bucket on the agg's own sweep, not a config map
    ck2val = {} if by_sweep else _cfgkey_value_map(scoped, col)
    buckets: dict = {}
    for a, lab, sty in zip(chosen, labels, styles):
        val = a["sweep_name"] if by_sweep else ck2val.get(a["config_key"])
        buckets.setdefault(val, []).append((a, lab, sty))
    keys = sorted(buckets.keys(), key=lambda x: (x is None, str(x)))
    yr = _shared_y_range([[t[0] for t in buckets[k]] for k in keys], y_field)
    ncols = min(2, len(keys)) or 1
    grid = st.columns(ncols)
    for i, k in enumerate(keys):
        items = buckets[k]
        with grid[i % ncols]:
            shown = _sweep_tag(k) if by_sweep else k
            st.markdown(f"**{group_label} = {shown}**")
            fig = _build_comparison_chart(
                [t[0] for t in items], x_field, y_field,
                short_labels=[t[1] for t in items],
                styles=[t[2] for t in items], y_range=yr)
            st.plotly_chart(fig, width="stretch", key=f"facet_{group_label}_{k}_{i}")


def _render_comparison():
    st.header("train_al comparison")
    if df.empty:
        st.info("No result.json files found under the output directory.")
        return

    cat_map = _registry_category_map()
    entry_map = _registry_entry_map()

    def _flags_of(sid):
        f = entry_map.get(str(sid), {}).get("flags")
        return f if isinstance(f, list) else []

    def _cat_for(row):
        return cat_map.get(
            str(row["sweep_id"]),
            reg.categorize(row.get("sweep_name", ""), row.get("command", ""))["category"],
        )

    gdf = df.copy()
    gdf["category"] = gdf.apply(_cat_for, axis=1)
    qp_out: dict = {}  # accumulates URL state to write back at the end

    # ── Category scope (first filter — prevents mixing unrelated curves) ──
    cat_opts = ["(all)"] + sorted(gdf["category"].unique())
    cats_present = cat_opts[1:]
    sel_cat = st.selectbox(
        "Category", options=cat_opts,
        index=_qp_index(cat_opts, _qp_get("cat"), 1 if cats_present else 0),
        key="cmp_cat")
    scoped = gdf if sel_cat == "(all)" else gdf[gdf["category"] == sel_cat]

    # ── Hide flagged sweeps by default (Feature 1) — BEFORE the sweep filter,
    #    so deprecated/buggy sweeps don't even appear in the picker unless shown.
    show_hidden = st.checkbox(
        "Show deprecated/buggy sweeps", value=(_qp_get("showhidden") == "1"),
        key="cmp_showhidden",
        help="Flagged sweeps are hidden by default; when shown they render dashed + grey.")
    if not show_hidden:
        hidden_ids = {sid for sid in scoped["sweep_id"].astype(str).unique()
                      if set(_flags_of(sid)) & reg.HIDDEN_FLAGS}
        if hidden_ids:
            scoped = scoped[~scoped["sweep_id"].astype(str).isin(hidden_ids)]

    # ── Sweep filter (narrow to specific sweeps; shows curated titles) ───────
    sweep_names = sorted(scoped["sweep_name"].dropna().unique().tolist())
    title_map = {e.get("sweep_name"): e.get("title", "")
                 for e in reg.load_registry() if e.get("title")}

    def _sweep_label(s):
        t = title_map.get(s)
        return f"{t}  ·  {s}" if t else s

    sweep_seed = [s for s in _qp_list("sweeps") if s in sweep_names]
    sel_sweeps = st.multiselect(
        "Sweep(s)  (empty = all)", options=sweep_names, default=sweep_seed,
        format_func=_sweep_label, key="cmp_sweeps")
    qp_out["sweeps"] = ",".join(sel_sweeps)
    if sel_sweeps:
        scoped = scoped[scoped["sweep_name"].isin(sel_sweeps)]

    # ── Date range ───────────────────────────────────────────────────────
    # An explicit sweep selection is the stronger intent than a coarse date
    # window, so when sweeps are named the date filter is NOT applied (otherwise
    # a sweep run outside the window would silently vanish — e.g. a 06-04 sweep
    # dropped by dhi=06-03). We also clear any stale date params from the URL.
    if sel_sweeps:
        if _qp_get("dlo") or _qp_get("dhi"):
            qp_out["dlo"] = qp_out["dhi"] = ""
            st.caption("📅 Date range ignored — specific sweeps are selected above.")
    elif "created_at" in scoped.columns and scoped["created_at"].notna().any():
        dmin = scoped["created_at"].min().date()
        dmax = scoped["created_at"].max().date()

        def _clamp_date(s, fallback):
            if not s:
                return fallback
            ts = pd.Timestamp(s)
            if pd.isna(ts):
                return fallback
            return min(max(ts.date(), dmin), dmax)

        dlo0 = _clamp_date(_qp_get("dlo"), dmin)
        dhi0 = _clamp_date(_qp_get("dhi"), dmax)
        dr = st.date_input("Date range", value=(dlo0, dhi0),
                           min_value=dmin, max_value=dmax, key="cmp_date")
        if isinstance(dr, tuple) and len(dr) == 2:
            lo, hi = pd.Timestamp(dr[0]), pd.Timestamp(dr[1]) + pd.Timedelta(days=1)
            scoped = scoped[(scoped["created_at"] >= lo) & (scoped["created_at"] < hi)]
            qp_out["dlo"] = "" if dr[0] == dmin else dr[0].isoformat()
            qp_out["dhi"] = "" if dr[1] == dmax else dr[1].isoformat()

    # ── Dynamic param filters: only params that vary inside the scope ─────
    # NOTE: p_seed is intentionally NOT here — curves are seed-averaged, so seed
    # is handled by the dedicated "Seeds" selector below (average vs individual).
    FILTERABLE = [
        "p_al_method", "p_tau", "p_target_nu", "p_budget", "p_dataset",
        "p_model", "p_eval_model_type", "p_features_tag", "p_max_iter",
        "p_kernel_type", "p_diff_method", "p_alpha", "p_initial_delta",
        "p_sparse_K", "status",
    ]
    candidates = [c for c in FILTERABLE if c in scoped.columns and scoped[c].nunique(dropna=True) > 1]
    if candidates:
        st.markdown("**Filters**")
        cols = st.columns(min(4, len(candidates)))
        for i, col in enumerate(candidates):
            vals = sorted(scoped[col].dropna().unique().tolist(), key=str)
            seed = set(_qp_list(f"f_{col}"))
            default = [v for v in vals if str(v) in seed]
            pick = cols[i % len(cols)].multiselect(
                col[2:] if col.startswith("p_") else col, options=vals,
                default=default, key=f"flt_{col}")
            qp_out[f"f_{col}"] = ",".join(str(v) for v in pick)
            if pick:
                scoped = scoped[scoped[col].isin(pick)]

    # ── Aggregate + chart (reuses scanner pipeline) ──────────────────────
    agg_list = [a for a in get_all_aggregated(scoped) if a["n_seeds"] > 0]
    if not agg_list:
        st.info("No runs with recorded episodes match the current filters.")
        return

    short = _labels_with_sweep(agg_list)
    seen: dict[str, int] = {}
    disp = []
    for sl, a in zip(short, agg_list):
        if short.count(sl) > 1:
            sl = f"{a['sweep_name']} / {sl}"
        if sl in seen:
            sl = f"{sl} [{a['config_key']}]"
        seen[sl] = 1
        disp.append(sl)
    opts = dict(zip(disp, agg_list))
    # Curves are serialised by stable config_key (display labels drift with the
    # diff set), so a shared link survives new sweeps landing.
    label_to_key = {lbl: a["config_key"] for lbl, a in opts.items()}
    key_to_label = {v: k for k, v in label_to_key.items()}

    # All curves are selected BY DEFAULT, and re-selected whenever the available
    # set changes (a new filter/sweep choice) — so newly-matching curves show up
    # plotted, not silently unchecked. We manage cmp_curves in session_state
    # (set before the widget, never after) to make this robust with the keyed
    # multiselect; the URL seeds the very first render.
    cur_sig = tuple(sorted(label_to_key.values()))
    if "cmp_curves" not in st.session_state:
        seed_keys = _qp_list("curves")
        st.session_state["cmp_curves"] = (
            [key_to_label[k] for k in seed_keys if k in key_to_label]
            if seed_keys else list(disp))
    elif st.session_state.get("cmp_curves_sig") != cur_sig:
        st.session_state["cmp_curves"] = list(disp)   # set changed → select all
    else:
        # Keep selection valid against the current options.
        st.session_state["cmp_curves"] = [
            s for s in st.session_state["cmp_curves"] if s in disp]
    st.session_state["cmp_curves_sig"] = cur_sig

    bsa, bcl, _ = st.columns([1, 1, 6])
    if bsa.button("Select all", key="cmp_selall"):
        st.session_state["cmp_curves"] = list(disp)
        st.rerun()
    if bcl.button("Clear", key="cmp_clearall"):
        st.session_state["cmp_curves"] = []
        st.rerun()
    sel = st.multiselect("Curves to plot", options=disp, key="cmp_curves")
    # Omit from the URL when all are selected (the default) → short shareable links.
    qp_out["curves"] = ("" if set(sel) == set(disp)
                        else ",".join(label_to_key[s] for s in sel))
    if not sel:
        st.info("Select at least one curve.")
        _qp_write(qp_out)
        return
    chosen = [opts[k] for k in sel]
    labels = _labels_with_sweep(chosen)
    styles = [{"flagged": bool(set(_flags_of(a["sweep_id"])) & reg.HIDDEN_FLAGS)}
              for a in chosen]

    # ── Seeds: 'average' (default, seed-mean) and/or individual seed trends ──
    seed_vals = (sorted(scoped["p_seed"].dropna().astype(int).unique().tolist())
                 if "p_seed" in scoped.columns else [])
    seed_opts = ["average"] + [str(s) for s in seed_vals]
    if len(seed_vals) > 1:
        sv_seed = [m for m in _qp_list("sv") if m in seed_opts] or ["average"]
        sel_seedmodes = st.multiselect(
            "Seeds  ·  'average' = mean across seeds; add individual seeds to overlay their trend",
            options=seed_opts, default=sv_seed, key="cmp_seeds") or ["average"]
    else:
        sel_seedmodes = ["average"]
    qp_out["sv"] = "" if sel_seedmodes == ["average"] else ",".join(sel_seedmodes)

    # Expand each chosen (seed-averaged) curve into the requested seed views.
    # 'average' reuses the seed-mean agg; an individual seed is aggregated on the
    # fly from that one seed's run (n_seeds=1, no ±std band).
    show_seed = sel_seedmodes != ["average"]
    r_aggs, r_labels, r_styles = [], [], []
    for a, lab, sty in zip(chosen, labels, styles):
        for mode in sel_seedmodes:
            if mode == "average":
                ra, tag = a, "avg"
            else:
                sub = scoped[(scoped["config_key"] == a["config_key"])
                             & (scoped["sweep_id"].astype(str) == str(a["sweep_id"]))
                             & (scoped["p_seed"] == int(mode))]
                saggs = [s for s in get_all_aggregated(sub) if s["n_seeds"] > 0]
                if not saggs:
                    continue
                ra, tag = saggs[0], f"seed{mode}"
            r_aggs.append(ra)
            r_labels.append((f"{lab} · {tag}" if lab else tag) if show_seed else lab)
            r_styles.append(sty)
    if not r_aggs:
        st.info("No data for the selected seed(s).")
        _qp_write(qp_out)
        return

    # ── Axes (Feature 2) + split-by (Feature 3) ──────────────────────────
    ac1, ac2, ac3 = st.columns(3)
    x_opts = list(_X_AXES.keys())
    x_label = ac1.selectbox("X-axis", x_opts,
                            index=_qp_index(x_opts, _qp_get("x")), key="cmp_x")
    x_field = _X_AXES[x_label]
    y_opts = _y_axis_options(chosen)
    y_field = ac2.selectbox("Y-axis", y_opts,
                            index=_qp_index(y_opts, _qp_get("y")), key="cmp_y",
                            format_func=lambda y: _Y_TITLE.get(y, y))
    group_opts = ["(none)"] + _groupable_options(scoped, chosen)
    group_label = ac3.selectbox("Split into separate graphs by", group_opts,
                                index=_qp_index(group_opts, _qp_get("group")),
                                key="cmp_group")
    qp_out.update(x=x_label, y=y_field,
                  group="" if group_label == "(none)" else group_label,
                  cat="" if sel_cat == "(all)" else sel_cat,
                  showhidden="1" if show_hidden else "")
    if x_field == "budget_units":
        st.caption("ℹ️ Oracle budget units = #labeled examples × τ "
                   "(oracle τ=1 → equals #examples).")

    if group_label == "(none)":
        fig = _build_comparison_chart(r_aggs, x_field, y_field,
                                      short_labels=r_labels, styles=r_styles)
        st.plotly_chart(fig, width="stretch")
    else:
        _render_facets(scoped, r_aggs, r_labels, r_styles, x_field, y_field, group_label)

    rows = [{
        "Config": sl, "Seeds": a["n_seeds"],
        "Final Acc": f"{a['final_test_acc_mean']:.2f}%" if a["final_test_acc_mean"] is not None else "—",
        "± Std": f"{a['final_test_acc_std']:.2f}%" if a["final_test_acc_std"] is not None else "—",
        "Flags": ", ".join(_flags_of(a["sweep_id"])) or "—",
    } for a, sl in zip(r_aggs, r_labels)]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── About this curve (Feature 0 read-only card) + Reproduce (Feature 5a) ──
    st.divider()
    st.markdown("#### ℹ️ About this curve")
    about_seed = key_to_label.get(_qp_get("about"))
    about_sel = st.selectbox(
        "Curve", options=sel,
        index=_qp_index(sel, about_seed if about_seed in sel else None), key="cmp_about")
    about_agg = opts[about_sel]
    sid = about_agg["sweep_id"]
    sweep_df = scoped[scoped["sweep_id"].astype(str) == str(sid)]
    _render_sweep_card(sid, sweep_df, editable=False,
                       key_prefix=f"about_{about_agg['config_key']}")
    # Reproduce uses THIS curve's exact config command (not the sweep's latest).
    cfg_df = scoped[scoped["config_key"] == about_agg["config_key"]]
    _render_reproduce(describe_sweep(cfg_df), key_prefix=f"about_{about_agg['config_key']}")
    qp_out["about"] = about_agg["config_key"]

    # ── Persist the view to the URL + share link (Feature 4) ─────────────
    _qp_write(qp_out)
    st.divider()
    st.caption(
        "🔗 **Shareable view** — this page's filters, axes, grouping and selection "
        "are encoded in the browser URL above. Copy the address bar to share or "
        "bookmark this exact view."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL DATA (ν' explorer + artifact browser)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_additional():
    st.header("Additional data")
    sub_nu, sub_art = st.tabs(["ν' cache", "Artifact browser"])

    # ── ν' explorer ──────────────────────────────────────────────────────────
    with sub_nu:
        nu_dataset = st.selectbox("Dataset", ["cifar100", "cifar10", "imagenet50"], key="nx_ds")
        cache = _load_nu_cache(nu_dataset)
        if not cache:
            st.warning(f"No ν' cache for {nu_dataset}. Run `sweep_nu_prime.py --submit`.")
        else:
            methods = sorted(cache.keys())
            budgets = sorted({int(b) for m in cache.values() for b in m})
            c1, c2 = st.columns(2)
            sm = c1.multiselect("AL methods", methods, default=methods, key="nx_m")
            sb = c2.multiselect("Budgets", budgets, default=budgets, key="nx_b")
            anchor = st.checkbox("Oracle anchor (τ=1, ν'=0)", value=True, key="nx_a")
            if sm and sb:
                fig = go.Figure()
                for mi, m in enumerate(sm):
                    for bi, b in enumerate(sb):
                        td = cache.get(m, {}).get(str(b))
                        if not td:
                            continue
                        taus = sorted(float(t) for t in td)
                        nus = [td[f"{t:.4f}"] for t in taus]
                        if anchor and 1.0 not in taus:
                            taus.append(1.0); nus.append(0.0)
                        fig.add_trace(go.Scatter(
                            x=taus, y=nus, mode="lines+markers", name=f"{m} / B={b}",
                            line=dict(color=_COLORS[mi % len(_COLORS)], width=2.2,
                                      dash=["solid", "dash", "dot", "dashdot"][bi % 4]),
                            marker=dict(size=7)))
                fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
                fig.update_layout(
                    xaxis_title="τ", yaxis_title="ν'", xaxis=dict(range=[0, 1.05]),
                    yaxis=dict(range=[0, 1.0]), template="plotly_white", height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    title=f"{nu_dataset.upper()} — ν' vs τ")
                st.plotly_chart(fig, width="stretch")

    # ── Generic artifact browser over results/** ──────────────────────────────
    with sub_art:
        st.caption("Renders .png / .csv / .json under results/. New files appear automatically.")
        if not _RESULTS_DIR.exists():
            st.info("No results/ directory.")
        else:
            arts = sorted(
                [p for p in _RESULTS_DIR.rglob("*")
                 if p.suffix.lower() in (".png", ".csv", ".json")],
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not arts:
                st.info("No artifacts found.")
            else:
                rel = [str(p.relative_to(_RESULTS_DIR)) for p in arts]
                pick = st.selectbox(f"Artifact ({len(arts)})", options=rel)
                path = _RESULTS_DIR / pick
                suf = path.suffix.lower()
                if suf == ".png":
                    st.image(str(path), caption=pick, width="stretch")
                elif suf == ".csv":
                    cdf = pd.read_csv(path)
                    st.dataframe(cdf, width="stretch", hide_index=True)
                    numcols = cdf.select_dtypes("number").columns.tolist()
                    if len(numcols) >= 2:
                        xx = st.selectbox("X", numcols, key="art_x")
                        yy = st.multiselect("Y", [c for c in numcols if c != xx],
                                            default=[c for c in numcols if c != xx][:1], key="art_y")
                        if yy:
                            f = go.Figure()
                            for c in yy:
                                f.add_trace(go.Scatter(x=cdf[xx], y=cdf[c], mode="lines+markers", name=c))
                            f.update_layout(xaxis_title=xx, template="plotly_white", height=420)
                            st.plotly_chart(f, width="stretch")
                elif suf == ".json":
                    try:
                        st.json(json.loads(path.read_text()))
                    except (json.JSONDecodeError, OSError) as exc:
                        st.error(f"Could not read JSON: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHS › NPC REPRO  (Netta's noise-aware ProbCover: NPC vs ProbCover vs Random,
#                      one panel per noise level)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_npc():
    st.header("NPC Reproduction — Noise-aware ProbCover")
    st.caption(
        "Netta's thesis: under a noisy annotator, **NPC** should beat plain "
        "**ProbCover** (which keeps 'covering' mislabeled points) and **Random**, "
        "with the gap widening as label noise grows. Test accuracy vs labeled-set "
        "size, seed-averaged with ±1 std bands, one panel per noise level."
    )
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch sub-tab.")
        return
    # Auto-detect NPC sweeps: those that contain an `npc` AL method.
    npc_sweeps = sorted(
        df[df["p_al_method"] == "npc"]["sweep_name"].dropna().unique().tolist()
    )
    if not npc_sweeps:
        st.info(
            "No NPC runs found (no run with al_method=npc). Launch the "
            "Netta sweep, e.g. `batch_utils/launch_netta_npc_sweep.py`."
        )
        return
    sel_sweeps = st.multiselect(
        "NPC sweep(s)", options=npc_sweeps, default=npc_sweeps,
        key="npc_sweep_sel",
    )
    x_axis_npc = st.radio(
        "X-axis", ["Labeled Set Size", "Episode"], horizontal=True,
        key="npc_xaxis",
    )
    sub = df[df["sweep_name"].isin(sel_sweeps)]
    agg_list = get_all_aggregated(sub)

    # Bucket: noise_level → method → agg dict
    panels: dict[int, dict] = {}
    misc = []  # configs we couldn't classify (no NETTA_n tag)
    for agg in agg_list:
        lbl = agg["config_label"]
        noise = _npc_noise_of(lbl)
        method = _npc_method_of(lbl)
        if noise is None or method not in _NPC_METHOD_ORDER:
            misc.append(agg)
            continue
        panels.setdefault(noise, {})[method] = agg

    if not panels:
        st.warning("Selected sweep(s) have no NETTA_n{20,50,80} configs to plot.")
        return
    noise_levels = sorted(panels.keys())
    cols = st.columns(len(noise_levels))
    for col, noise in zip(cols, noise_levels):
        with col:
            st.subheader(f"{noise}% label noise")
            fig = _build_npc_panel(panels[noise], x_axis_npc)
            st.plotly_chart(fig, width="stretch", key=f"npc_panel_{noise}")

    # Final-accuracy summary table (last-episode mean ± std).
    rows = []
    for noise in noise_levels:
        for method in _NPC_METHOD_ORDER:
            agg = panels[noise].get(method)
            if not agg or agg["final_test_acc_mean"] is None:
                continue
            rows.append({
                "noise %": noise,
                "method": _NPC_METHOD_LABEL[method],
                "final acc (%)": round(agg["final_test_acc_mean"], 2),
                "± std": round(agg["final_test_acc_std"], 2),
                "seeds": agg["n_seeds"],
            })
    if rows:
        st.markdown("**Final-episode test accuracy**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if misc:
        st.caption(
            f"{len(misc)} config(s) in the selected sweep(s) were not "
            "NETTA_n{20,50,80}/npc/probcover/random and are omitted here "
            "(see the Comparison sub-tab)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHS › RUN DETAILS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_details():
    st.header("Run Details")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch sub-tab.")
        return
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
        return
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
            pd.DataFrame(list(params_dict.items()), columns=["Parameter", "Value"]),
            width="stretch", hide_index=True,
        )

    # Episode table + chart
    episodes = run.get("episodes") or []
    if episodes:
        st.subheader("Per-Episode Results")
        ep_df = pd.DataFrame(episodes)
        ep_rename = {
            "episode": "Episode", "test_acc": "Test Acc (%)",
            "val_acc": "Val Acc (%)", "train_acc": "Train Acc (%)",
            "labeled_size": "Labeled Size", "duration_s": "Duration (s)",
        }
        st.dataframe(ep_df.rename(columns=ep_rename), width="stretch", hide_index=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ep_df["episode"], y=ep_df["test_acc"], mode="lines+markers",
            name="Test Acc", line=dict(color=_COLORS[0], width=2)))
        if "val_acc" in ep_df.columns:
            fig2.add_trace(go.Scatter(
                x=ep_df["episode"], y=ep_df["val_acc"], mode="lines+markers",
                name="Val Acc", line=dict(color=_COLORS[1], width=2, dash="dash")))
        fig2.update_layout(
            title=f"Accuracy — seed={seed_sel}", xaxis_title="Episode",
            yaxis_title="Accuracy (%)", template="plotly_white", height=350,
            margin=dict(t=40))
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No episode data recorded yet (run may still be starting).")

    # Log viewer
    st.subheader("stdout.log")
    log_path = Path(run["exp_dir"]) / "stdout.log"
    if log_path.exists():
        filter_str = st.text_input(
            "Filter lines (case-insensitive substring)", key="log_filter")
        try:
            log_text = log_path.read_text(errors="replace")
            if filter_str:
                lines = [ln for ln in log_text.splitlines()
                         if filter_str.lower() in ln.lower()]
                log_text = "\n".join(lines)
            st.text_area("log", log_text, height=420, label_visibility="collapsed")
        except OSError as exc:
            st.warning(f"Could not read log: {exc}")
    else:
        st.info(f"Log not found at `{log_path}`")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHS › EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _render_export():
    st.header("Export Report")
    if df.empty:
        st.info("No results yet. Start a sweep from the 🚀 Launch sub-tab.")
        return
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
            "⬇ Download report.html", data=html_content,
            file_name="al_report.html", mime="text/html", width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEPS › LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════

def _render_launch():
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
streamlit run reporting/dashboard.py --server.port 8501
```
Then on your laptop open the SSH tunnel:
```bash
ssh -L 8501:localhost:8501 alongol@takara
```
And browse to **http://localhost:8501**.
            """
        )

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

        if not key_path.exists():
            r = subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                return False, f"ssh-keygen failed: {r.stderr.strip()}"

        if not pub_path.exists():
            return False, f"Public key not found at `{pub_path}`."

        pub_key_line = pub_path.read_text().strip()

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
            sftp.chmod(ak_path, 0o600)
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
        _slurm_node = 'phoenix-gw'
        _default_al = ["--al probcover"]
        _default_grid = {"seed": [1, 2, 3]}
        _default_fixed = ""

    # ── Config editor ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        al_text = st.text_area(
            "AL\\_CONFIGS — one variant per line",
            value="\n".join(_default_al), height=170,
            help="Each line is one AL experiment variant. Lines are NOT crossed with each other.",
            key="launch_al_configs",
        )
        fixed_args_input = st.text_input(
            "FIXED\\_ARGS (added to every job)",
            value=_default_fixed, key="launch_fixed_args",
        )

    with col_right:
        _grid_default_text = "\n".join(
            f"{k}={','.join(str(v) for v in vals)}"
            for k, vals in _default_grid.items()
        )
        grid_text = st.text_area(
            "GRID — key=val1,val2,... per line (fully crossed with each AL\\_CONFIG)",
            value=_grid_default_text, height=170,
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
                al_configs_parsed, grid_parsed, fixed_args_input)
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
        if st.button("▶ Run locally", disabled=not parsed_ok or not preview_jobs,
                     width="stretch", key="btn_local"):
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
            st.info("📋 Switch to the **Ledger** sub-tab — your new sweep appears there.")
            st.cache_data.clear()

    with col_sbatch:
        st.markdown(f"**📤 Submit to SLURM** — via `{_slurm_node}`")
        _ssh_ok, _ssh_msg = _check_ssh_auth(_slurm_node)
        if _ssh_ok:
            st.caption(_ssh_msg)
        else:
            st.warning(f"No passwordless SSH to `{_slurm_node}` yet — SLURM submission will fail.")
            _ssh_password = st.text_input(
                f"Your password for `{_slurm_node}`", type="password",
                key="ssh_password_input",
                help="Used once to copy your SSH key. Never stored anywhere.",
            )
            if st.button("🔑 Set up passwordless SSH now", key="btn_setup_ssh",
                         width="stretch", disabled=not _ssh_password):
                with st.spinner(f"Copying SSH key to {_slurm_node}…"):
                    _setup_ok, _setup_msg = _setup_ssh_key(_slurm_node, _ssh_password)
                if _setup_ok:
                    st.success(_setup_msg)
                else:
                    st.error(_setup_msg)
        if st.button("📤 Submit to SLURM", disabled=not parsed_ok or not preview_jobs,
                     width="stretch", key="btn_sbatch"):
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
        "🗑 Clear done", key="clear_done_launches",
        help="Remove all non-running launches from this list",
    ):
        _launches = [l for l in _launches if _is_process_running(l.get("pid"))]
        _save_launches(_launches)
        st.rerun()

    if not _launches:
        st.info("No launches yet. Use the buttons above to start your first sweep.")
        return
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
                    "Lines to show", min_value=20, max_value=1000, value=60,
                    key=f"tail_{_launch.get('sweep_id', id(_launch))}",
                )
                _log_lines = _log_path.read_text(errors="replace").splitlines()
                st.text_area(
                    "log", value="\n".join(_log_lines[-int(_tail_n):]),
                    height=380, key=f"log_{_launch.get('sweep_id', id(_launch))}",
                    label_visibility="collapsed",
                )
                st.caption(f"`{_log_path}`")
                if st.button("🔄 Refresh log",
                             key=f"refresh_{_launch.get('sweep_id', id(_launch))}"):
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


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — segmented-control nav (URL-synced), with per-section sub-nav.
#
# We use st.segmented_control rather than st.tabs because native tabs are
# client-side only — Streamlit can't report or set the active tab, so it can't be
# put in the URL. Segmented controls are keyed widgets: the active tab + sub-tab
# serialise into st.query_params (`tab`, `sub`) and a shared link reopens them.
# ═══════════════════════════════════════════════════════════════════════════════


def _nav(label: str, qp_key: str, options: dict, widget_key: str) -> str:
    """A URL-synced segmented-control 'tab bar'. options = {display_label: code}.
    Returns the selected display label."""
    labels = list(options)
    codes = list(options.values())
    seeded = _qp_get(qp_key)
    default = labels[codes.index(seeded)] if seeded in codes else labels[0]
    sel = st.segmented_control(
        label, labels, default=default, key=widget_key,
        label_visibility="collapsed")
    if sel is None:  # segmented_control returns None if the user clears it
        sel = default
    # Omit the default (first) option from the URL → clean links.
    _qp_write({qp_key: "" if sel == labels[0] else options[sel]})
    return sel


_TOP = {"📋 Sweeps": "sweeps", "📈 Graphs (train_al)": "graphs",
        "📊 Additional data": "extra"}
top_sel = _nav("Section", "tab", _TOP, "nav_top")

if top_sel == "📋 Sweeps":
    sub = _nav("Sweeps section", "sub",
               {"📒 Ledger": "ledger", "🚀 Launch": "launch"}, "nav_sweeps")
    if sub == "📒 Ledger":
        _render_ledger()
    else:
        _render_launch()

elif top_sel == "📈 Graphs (train_al)":
    sub = _nav("Graphs section", "sub",
               {"📈 Comparison": "cmp", "🧪 NPC Repro": "npc",
                "🔍 Run Details": "details", "📤 Export": "export"}, "nav_graphs")
    if sub == "📈 Comparison":
        _render_comparison()
    elif sub == "🧪 NPC Repro":
        _render_npc()
    elif sub == "🔍 Run Details":
        _render_details()
    else:
        _render_export()

else:
    _render_additional()

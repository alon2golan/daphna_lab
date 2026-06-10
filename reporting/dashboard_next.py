"""
AL Dashboard (next) — reorganised, runs SIDE-BY-SIDE with the current dashboard.

    streamlit run reporting/dashboard.py                       # current (port 8501)
    streamlit run reporting/dashboard_next.py --server.port 8502   # this one

Three tabs, by data source:
  1. 📋 Sweeps        – general SLURM job ledger (ALL kinds), from the registry
                        (logs/sweep/sweep_registry.json). Grouped by category;
                        per-sweep category override. Seed it via
                        `python reporting/backfill_registry.py`.
  2. 📈 Graphs        – train_al experiment comparison (result.json), with the
                        full filter set (date, al_method, τ, ν, B, …).
  3. 📊 Additional    – ν' cache explorer + a generic artifact browser over
                        results/** (renders .png / .csv / .json).

Nothing here mutates the current dashboard; it only reads shared modules.
"""

from __future__ import annotations

import json
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
    get_all_aggregated,
    load_all_results,
)
from reporting import registry as reg  # noqa: E402
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

with st.sidebar:
    st.title("🧪 AL Dashboard (next)")
    output_dir = st.text_input("Output directory", value=str(REPO_ROOT / "output"))
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Parallel to the current dashboard. Auto-refreshes every 30 s.")


# ── Data loading ────────────────────────────────────────────────────────────────


@st.cache_data(ttl=30, show_spinner="Scanning result.json…")
def _load_results(directory: str) -> pd.DataFrame:
    return load_all_results(directory)


def _registry_category_map() -> dict[str, str]:
    return {e["sweep_id"]: e.get("category", "uncategorized") for e in reg.load_registry()}


df = _load_results(output_dir)


# ── Shared chart helpers (copied from dashboard.py so this file is standalone) ──


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


def _build_comparison_chart(agg_list, x_axis="Episode", short_labels=None) -> go.Figure:
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
        label = (short_labels[i] if short_labels else None) or \
            f"{agg['sweep_name']} / {agg['config_label']}"
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1], fill="toself", fillcolor=color,
            opacity=0.15, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines+markers", name=f"{label} (n={agg['n_seeds']})",
            line=dict(color=color, width=2.5), marker=dict(size=7),
            hovertemplate=f"<b>{label}</b><br>" +
            ("Episode: %{x}" if x_axis == "Episode" else "Labeled: %{x}") +
            "<br>Test Acc: %{y:.2f}%<extra></extra>"))
    fig.update_layout(
        xaxis_title=x_axis, yaxis_title="Test Accuracy (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", template="plotly_white", height=500, margin=dict(t=40))
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_sweeps, tab_graphs, tab_extra = st.tabs(
    ["📋 Sweeps", "📈 Graphs (train_al)", "📊 Additional data"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SWEEPS LEDGER (registry-backed; all job kinds)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sweeps:
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
        all_cats = sorted(reg_df["category"].unique())
        all_kinds = sorted(reg_df["kind"].unique())

        fc1, fc2, fc3 = st.columns([2, 2, 1])
        sel_cats = fc1.multiselect("Category (empty = all)", options=all_cats)
        sel_kinds = fc2.multiselect("Kind (empty = all)", options=all_kinds)
        if fc3.button("🔁 Rescan SLURM", help="Re-run backfill incl. one-shot sacct"):
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

        st.write(f"**{len(view)}** sweeps")

        # Group by category, known order first.
        ordered_cats = [c for c in reg.KNOWN_CATEGORIES if c in view["category"].unique()]
        ordered_cats += [c for c in sorted(view["category"].unique()) if c not in ordered_cats]

        for cat in ordered_cats:
            cat_rows = view[view["category"] == cat]
            st.subheader(f"{cat}  ·  {len(cat_rows)}")
            for ridx, (_, e) in enumerate(cat_rows.iterrows()):
                sid = e["sweep_id"]
                p = prog.get(sid)
                if p:
                    badge = f"✅{p['completed']} 🔄{p['running']} ❌{p['failed']} / {p['total']}"
                else:
                    badge = f"_{e['kind']}_"
                date = (e.get("created_at") or "")[:16].replace("T", " ")
                ukey = f"{sid}_{ridx}"
                with st.expander(f"**{e['sweep_name']}** — {badge}  ·  {date}"):
                    st.markdown(
                        f"**id:** `{sid}` &nbsp;|&nbsp; **kind:** `{e['kind']}` "
                        f"&nbsp;|&nbsp; **source:** `{e.get('source','?')}`"
                    )
                    if e.get("command"):
                        st.code(e["command"], language="bash")
                    if p:
                        st.progress(
                            p["completed"] / p["total"] if p["total"] else 0.0,
                            text=f"{p['completed']}/{p['total']} completed",
                        )
                    new_cat = st.selectbox(
                        "Category", options=reg.KNOWN_CATEGORIES,
                        index=reg.KNOWN_CATEGORIES.index(cat)
                        if cat in reg.KNOWN_CATEGORIES else 0,
                        key=f"cat_{ukey}",
                    )
                    if new_cat != cat:
                        reg.set_category(sid, new_cat)
                        st.rerun()
                    if e.get("note"):
                        st.caption(f"📝 {e['note']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRAPHS (train_al only; full filters)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graphs:
    st.header("train_al comparison")
    if df.empty:
        st.info("No result.json files found under the output directory.")
    else:
        # Attach category from the registry (fallback to auto-derive).
        cat_map = _registry_category_map()

        def _cat_for(row):
            return cat_map.get(
                str(row["sweep_id"]),
                reg.categorize(row.get("sweep_name", ""), row.get("command", ""))["category"],
            )

        gdf = df.copy()
        gdf["category"] = gdf.apply(_cat_for, axis=1)

        # ── Category scope (first filter — prevents mixing unrelated curves) ──
        cats_present = sorted(gdf["category"].unique())
        sel_cat = st.selectbox("Category", options=["(all)"] + cats_present, index=1 if cats_present else 0)
        scoped = gdf if sel_cat == "(all)" else gdf[gdf["category"] == sel_cat]

        # ── Date range ───────────────────────────────────────────────────────
        if "created_at" in scoped.columns and scoped["created_at"].notna().any():
            dmin = scoped["created_at"].min().date()
            dmax = scoped["created_at"].max().date()
            dr = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            if isinstance(dr, tuple) and len(dr) == 2:
                lo, hi = pd.Timestamp(dr[0]), pd.Timestamp(dr[1]) + pd.Timedelta(days=1)
                scoped = scoped[(scoped["created_at"] >= lo) & (scoped["created_at"] < hi)]

        # ── Dynamic param filters: only params that vary inside the scope ─────
        FILTERABLE = [
            "p_al_method", "p_tau", "p_target_nu", "p_budget", "p_dataset",
            "p_model", "p_eval_model_type", "p_features_tag", "p_max_iter",
            "p_kernel_type", "p_diff_method", "p_alpha", "p_initial_delta",
            "p_sparse_K", "status", "p_seed",
        ]
        candidates = [c for c in FILTERABLE if c in scoped.columns and scoped[c].nunique(dropna=True) > 1]
        if candidates:
            st.markdown("**Filters**")
            cols = st.columns(min(4, len(candidates)))
            for i, col in enumerate(candidates):
                vals = sorted(scoped[col].dropna().unique().tolist(), key=str)
                pick = cols[i % len(cols)].multiselect(
                    col[2:] if col.startswith("p_") else col, options=vals, key=f"flt_{col}")
                if pick:
                    scoped = scoped[scoped[col].isin(pick)]

        # ── Aggregate + chart (reuses scanner pipeline) ──────────────────────
        agg_list = [a for a in get_all_aggregated(scoped) if a["n_seeds"] > 0]
        if not agg_list:
            st.info("No runs with recorded episodes match the current filters.")
        else:
            short = _diff_labels(agg_list)
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
            sel = st.multiselect("Curves to plot", options=disp, default=disp)
            x_axis = st.radio("X-axis", ["Episode", "Labeled Set Size"], horizontal=True)
            if sel:
                chosen = [opts[k] for k in sel]
                labels = _diff_labels(chosen)
                fig = _build_comparison_chart(chosen, x_axis, short_labels=labels)
                st.plotly_chart(fig, width="stretch")
                rows = [{
                    "Config": sl, "Seeds": a["n_seeds"],
                    "Final Acc": f"{a['final_test_acc_mean']:.2f}%" if a["final_test_acc_mean"] is not None else "—",
                    "± Std": f"{a['final_test_acc_std']:.2f}%" if a["final_test_acc_std"] is not None else "—",
                } for a, sl in zip(chosen, labels)]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("Select at least one curve.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADDITIONAL DATA (ν' explorer + artifact browser)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_extra:
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

# Dashboard — Graphs section: feature plan

Status: **Features 0,1,2,3,4,5a SHIPPED** (2026-06-11) in `reporting/dashboard_next.py`
(+ `scanner.py`, `registry.py`). **5b (code snapshot + dependency vault) NOT yet built** —
design agreed (see "Feature 5b" below + the dependency-vault note). `dashboard.py` and the
`al-dashboard` skill left untouched for now (retire later).

Implemented this session (all headless-smoke-tested, incl. a URL round-trip test):
- **F0** description cards: registry `title`/`context`; `scanner.describe_sweep()` auto-facts;
  shared `_render_sweep_card()` (editable in Ledger, read-only "About this curve" in Graphs).
- **F1** flags: registry `flags` list + `set_flags()`; `FLAG_VOCAB={deprecated,bug}`,
  `HIDDEN_FLAGS`; Ledger buttons (🐞/🗑/↩) + note; hidden-by-default with a show-toggle in both
  Ledger and Comparison; flagged curves render dashed+grey.
- **F2** axes: `aggregate_config_episodes` widened to carry every numeric per-episode field as
  `<f>_mean/_std` + `labeled_size` + derived `budget_units = labeled_size × τ`. X∈{episode,
  #examples, budget units}, Y discovered from `*_mean` cols. `tau` + `sweep_id` now on each agg.
- **F3** facets: `_render_facets()` + `_shared_y_range()`; "Split into separate graphs by"
  selectbox over varying params/category, 2-col grid, shared Y range.
- **F5a** reproduce: `_render_reproduce()` shows the curve's exact command + commit + exp_dir.
- **F4** URL state: `_qp_get/_qp_list/_qp_index/_qp_write`; cat/x/y/group/curves(by config_key)/
  about/showhidden/date/filters encoded in `st.query_params`; index/default-from-qp pattern (no
  session_state writes → no rerun loops); curves omitted from URL when all selected.
- `registry.py` CLI gained `--edit/--title/--context/--flag/--unflag`. `backfill_registry`
  preserves all new fields automatically (via `upsert` + `_USER_FIELDS`).

Original target file: `reporting/dashboard_next.py` (plus `scanner.py`, `registry.py`,
`batch_utils/sweep_train_al.py`). The current `dashboard.py` is being retired in favour of
`dashboard_next.py`.

---

## Feature 5b — dependency vault (NEXT SESSION; design agreed with alon)

Resolved open questions:
- **Budget axis (F2):** budget_units = `labeled_size × τ`; #examples = `labeled_size` (oracle
  τ=1 → coincide). DONE.
- **Snapshot scope (5b):** include untracked **source** files but never `output/`/`results/` —
  free, because a temp-index `git add -A` respects `.gitignore` (which already excludes those).
- **The real coupling:** runs read `results/<ds>/nu_prime_cache.json` (`lookup_nu_for_tau`) and
  external feature files (`DATASET_FEATURES_DICT` in `deep_al/pycls/datasets/utils/features.py`,
  e.g. `/cs/labs/daphna/itai.david/representations_bank/...`). alon wants a **content-addressed,
  append-only, protected dependency vault** (never overwrite/remove; only add tags; verify by
  hash) so reproducibility never depends on the mutable `output/` folder.
- **Proposed vault** (`repro_store/` — location TBC with alon): `objects/<sha256>` immutable
  blobs (chmod 0444, deduped so a feature `.npy` is stored once), `tags/<name>.json`,
  `manifests/<sweep_id>.json` (code ref + [(logical_name, sha256)] for ν' cache + features).
  At launch: `git stash create` (+temp-index for untracked) → `refs/sweeps/<id>`; vault the ν'
  cache + resolved feature files; write manifest. Dashboard: extend the F5a panel with the code
  ref + `git diff vs base` + worktree cmd + per-dependency verify badge.
- Still open: vault path (in-repo vs sibling); store feature files (big, deduped) vs ν' cache
  only; auto-on-launch vs opt-in. New module `reporting/repro.py`; hook
  `launch_local_bg`/`submit_sbatch_jobs` in `sweep_train_al.py`.

## Vision

Re-frame the Graphs section as an explicit **pipeline**:

```
  all train_al runs
        │
   (1) FILTER      ── pick the runs you care about (params, date, status, not-deprecated)
        │
   (2) GROUP       ── optionally split into several panels (e.g. one per al_method)
        │
   (3) PLOT        ── choose X axis (episode / #examples / budget) and Y axis (acc / …),
        │              overlay the curves in each panel with ±std bands
        ▼
   every curve is DESCRIBED (a coherent card: date, params, context/story), TRACEABLE
   (exact command + code snapshot) and the whole view is SHAREABLE (one URL restores
   filters + grouping + axes + selection).
```

Six features deliver this. They are ordered **easiest → hardest**; each is shippable on
its own, and later features assume the earlier ones exist but don't require them.

---

## What we have today (facts the plan relies on)

- **Repo:** `daphna_lab` is a git repo; working branch `noise_removal_aum`. Streamlit 1.56.0.
- **Per-run dataframe** (`scanner.load_all_results`) columns: `exp_dir, result_file, status,
  created_at, finished_at, sweep_id, sweep_name, slurm_task_id, commit_hash, command,
  config_key, config_label, total_episodes_planned/done, final_test_acc, episodes, error,
  p_*` (every CLI param).
- **Per-episode fields** inside `episodes`: `episode, test_acc, val_acc, train_acc,
  labeled_size, duration_s`.
- **Aggregator** (`scanner.aggregate_config_episodes`) currently keeps only
  `episode, test_acc_mean, test_acc_std, val_acc_mean, labeled_size, n_seeds`.
  It **drops** `train_acc` and the std of val/train — Feature 2 must widen it.
- **Registry** (`reporting/registry.py`, `logs/sweep/sweep_registry.json`): one entry per
  sweep_id, freely extensible; the dashboard already writes back the `category` field, so
  adding a `flags`/`status` field is low-risk. `make_entry`/`upsert`/`set_category` exist.
- **Launch flow** (`sweep_train_al.py`): `launch_local_bg` (local detached bash) and
  `submit_sbatch_jobs` (sbatch, possibly via ssh). The dashboard Launch sub-tab calls these.

---

## Feature 0 (EASIEST) — Coherent description card per sweep (and per curve)

**Problem (user's new ask):** the runs are a blur — you don't remember *when* a sweep was
made, *what params* it used, or *why* it exists. Each sweep (and each graph/curve) should
carry a **coherent description**: date, params, and the **context/story** in your own words
(e.g. *"I fixed the ν' cache and re-ran the noisy_vs_oracle sweep"*).

**Design — split auto-facts from the human story** (so nothing goes stale):
- **Auto-derived facts** (computed live from `result.json` + git, never stored): date / date
  range, number of runs + status counts, the params that *vary* in the sweep
  (al_method, τ, ν, B, dataset, model, seeds…), the representative `command`, `commit_hash`,
  and `exp_dir`. These are always fresh, no duplication.
- **Curated story** (stored in the registry, edited by you): a short `title` and a multi-line
  markdown `context` — the *why*. This is a richer sibling of today's single-line `note`
  (keep `note` as a one-liner; `context` is the paragraph). The full **description card** =
  auto-facts table + rendered `context`.
- If no `title`/`context` is set yet, auto-suggest a starter from the sweep name + command, so
  even un-annotated sweeps render a sensible card.

**UX**
- **Ledger:** each sweep expander shows the card — editable `title` + `context` (markdown
  preview), plus the read-only auto-fact table (date, #runs, varying-params grid, command,
  commit, flags from Feature 1).
- **Graphs:** an **`ℹ️ About this curve`** panel surfaces the same card (read-only) for the
  selected curve/sweep — answering "what is this, when, which params, and why" without leaving
  the plot. (This is the natural home that Feature 5a's reproduce panel plugs into.)
- Editable from the dashboard **and** by asking me / CLI (same write-back path as flags), so
  "set the context of sweep 30797616 to 'ν′ cache dense sweep, PC+random'" is a one-liner.

**Implementation / files**
- `registry.py`: add `title` + `context` to `make_entry` (default `""`); preserve them in
  `upsert` (like `category`/`note`); add `set_description(sweep_id, title, context)`; extend
  the CLI (`--title`, `--context`).
- `scanner.py`: a small `describe_sweep(df_subset) -> dict` returning the auto-facts
  (date range, status counts, varying-param map, representative command/commit). Reused by
  both the Ledger card and the Graphs "About" panel.
- `dashboard_next.py`: a shared `_render_sweep_card(sweep_id, scope)` used in the Ledger
  (editable) and Graphs (read-only).
- `backfill_registry.py`: never wipe `title`/`context` (same rule as `note`).

**Effort:** S. **Risk:** low (additive registry fields + a read-only summariser).
**Open Q:** granularity — a description per *sweep* is clear; do you also want to annotate a
single *curve/config* (e.g. just the τ=0.1 curve within a sweep)? Proposal: sweep-level first
(1:1 with the registry), config-level context as a follow-up.

---

## Feature 1 (EASY) — Deprecate / flag sweeps, hidden by default

**Problem (user's #4):** when a sweep/graph turns out to be buggy or irrelevant, don't delete
it (we may want to reproduce it) — mark it, and hide it from the Graphs picker by default
while keeping it one toggle away.

**Design**
- Add an optional `flags` field to each registry entry: a list, e.g. `["deprecated"]`,
  `["bug"]`, or both, plus the existing free-text `note` for the reason. (A list, not a single
  status, so "bug" and "deprecated" can co-exist and we can add tags later.)
- **Filtered by default:** the Graphs filter and the Ledger hide any sweep whose `flags`
  intersect a "hidden" set (`{deprecated, bug}` by default). A `☑ Show deprecated/buggy`
  checkbox reveals them; when shown, their curves render dashed + greyed with a ⚠ in the label.
- **Two ways to set a flag:**
  1. *In the dashboard:* per-sweep buttons in the Ledger ("🐞 Mark bug", "🗑 Deprecate",
     "↩ Restore") + a note box; writes straight back to the registry.
  2. *By asking me / CLI:* extend `registry.py` CLI with `--flag bug --note "…"` /
     `--unflag`, so you can say "deprecate sweep 30811243 because X" and I edit the JSON.

**Implementation / files**
- `registry.py`: add `flags` to `make_entry` (default `[]`); preserve it in `upsert` (like
  `category`/`note`); add `set_flags(sweep_id, flags, note="")`; extend the CLI.
- `dashboard_next.py`: ledger gets the flag buttons; `_render_comparison` (and NPC) filter on
  flags + a "show hidden" checkbox; styling for shown-but-flagged curves.
- `backfill_registry.py`: ensure it never wipes `flags` (same rule as `note`).

**Effort:** S. **Risk:** very low (additive field, backward-compatible).
**Open Q:** fixed flag vocabulary `{bug, deprecated}` or free-form tags? (Proposal: start
fixed, the `flags` list leaves the door open.)

---

## Feature 2 (EASY) — Selectable X and Y axes

**Problem (user's #2):** different questions want different axes — X ∈ {episode, #labeled
examples, budget, …}, Y usually accuracy but sometimes val/train acc or other metrics.

**Design**
- **X axis** dropdown:
  - `episode` — 0,1,2,… (have it)
  - `# labeled examples` → `labeled_size` (have it)
  - `oracle budget` → **derived**: cumulative cost. Candidate formula
    `cumulative_budget = (episode + 1) * p_budget` (needs validation against how cost is
    actually accrued — see open Q). Flagged clearly until validated.
- **Y axis** dropdown: `test_acc` (default), `val_acc`, `train_acc`, `duration_s`,
  and whatever per-episode numeric fields exist — discovered dynamically so new logged
  metrics show up automatically.

**Implementation / files**
- `scanner.aggregate_config_episodes`: widen the aggregation to carry **every numeric
  per-episode field** as `<field>_mean`/`<field>_std` (not just test/val), and keep
  `labeled_size` + a derived `cumulative_budget`. This is the core change; it also benefits
  the NPC and Run-Details views.
- `dashboard_next.py`: `_build_comparison_chart` takes `(x_field, y_field)` instead of the
  current 2-way `x_axis` string; X/Y selectboxes drive it; axis titles + hovertemplate adapt.

**Effort:** S–M. **Risk:** low; the aggregator change is shared, so smoke-test NPC + Run
Details after. **Open Q:** exact budget/oracle-cost formula (does noise inflate it by 1/τ?
memory mentions a "3000 oracle-cost cap" — confirm the accounting before exposing it).

---

## Feature 3 (MEDIUM) — Filter → group-by into multiple graphs (facets)

**Problem (user's #3):** after filtering, sometimes overlay everything in one graph, sometimes
split — e.g. all `probcover` in one panel, all `random` in another.

**Design** — generalise the panel pattern already used by the NPC tab.
- After the filter step, a **`Split into separate graphs by`** selectbox:
  `(none)` | `al_method` | `tau` | `target_nu` | `budget` | `dataset` | `model` | `category` | …
  (offer any `p_*`/meta field that varies in the current selection).
- `(none)` → today's single overlaid chart. Otherwise → one panel per distinct value of the
  group key, each panel overlaying the curves that fall in it, laid out in a responsive grid
  (`st.columns`), shared Y range across panels for visual comparability.
- Optional **second-level group** (group by `al_method`, then within each panel colour by
  `tau`) — nice-to-have, can be a follow-up.

**Implementation / files**
- `dashboard_next.py`, `_render_comparison`: after building the curve list, bucket by the
  group key (reuse `_diff_labels` for in-panel labels), render N `_build_comparison_chart`
  figures. Factor the NPC panel/grid logic into a shared `_facet_grid(...)` helper so NPC and
  general comparison share one implementation.

**Effort:** M. **Risk:** low–medium (mostly layout/labels). Depends on Feature 2's axis
plumbing being in place so panels share axis choices.

---

## Feature 4 (MEDIUM–HARD) — Shareable URL state

**Problem (user's #5):** capture the exact view (filters + grouping + axes + selected curves
+ show-hidden) in the URL, so a link reproduces the page for you or a colleague.

**Design**
- Serialise the Graphs UI state into `st.query_params` on every change: `cat`, `x`, `y`,
  `group`, `flt_*` (each multiselect), `curves` (selected config keys), `showhidden`,
  `date_lo/hi`. Multi-values via comma-joined or repeated keys.
- On load, **seed widget defaults from the query params** (set `st.session_state` keys before
  the widgets are created), so the URL drives the initial render. A "🔗 Copy link" button
  surfaces the current full URL.
- Stable identity: curves keyed by `config_key` (stable) rather than display label (varies
  with the diff set), so links survive new sweeps landing.

**Implementation / files**
- `dashboard_next.py`: a small `url_state` helper (read → seed session_state → render widgets
  → write back). Care needed to avoid rerun loops (write only when changed) and to keep widget
  `key`s ↔ param names in sync.

**Effort:** M–H. **Risk:** medium — Streamlit state/query-param sync is fiddly (rerun loops,
type coercion, multiselect encoding). Self-contained to the Graphs tab; do it after 1–3 so
there's a stable set of controls to serialise.

---

## Feature 5 (HARDEST) — Per-sweep reproducibility snapshot (command + exact code)

**Problem (user's #1):** from a curve, know **exactly** how it was produced — the command AND
the code state — and be able to reproduce it. Snapshot the code before each sweep to a
**reproducible hash/tag**, but **not on the working branch**: the dev branch must behave as if
those files were never staged/committed.

Split into a free quick-win and the real (hard) part.

### 5a (S, quick win) — "Reproduce this curve" panel
We already store `command`, `commit_hash`, `exp_dir`, `sweep_id` per run. Add a "🔁 Reproduce"
section **inside Feature 0's "About this curve" panel** showing: the exact `command`, the
runtime `commit_hash`, and the `exp_dir`. Copy-paste ready. No new infra — it just extends the
description card with the how-to-rerun bits.

### 5b (H) — Pre-sweep working-tree snapshot to a side ref
**Goal:** capture the *exact* tree (including uncommitted edits) at launch, addressable by a
hash, without ever touching HEAD / the index / the working tree / the branch.

**Mechanism — `git stash create` + a private ref namespace:**
1. At launch (in `launch_local_bg` / `submit_sbatch_jobs`, on takara, before jobs run):
   - `snap=$(git stash create)` → builds commit objects for the current tree **without**
     modifying HEAD, index, working tree, or the stash list. If empty (no changes),
     fall back to `git rev-parse HEAD`.
   - `git update-ref refs/sweeps/<sweep_id> $snap` → parks the snapshot under a **custom ref
     namespace**. This ref is invisible to `git log`/`git status` of the branch, is **not**
     under `refs/heads`, and is **not pushed** by default → satisfies "not part of the main
     branch, behaves like I never staged it."
   - Record in the registry entry: `repro_snapshot` (sha), `repro_ref`
     (`refs/sweeps/<id>`), `base_commit` (HEAD sha), `branch`, `dirty` (bool).
2. **SLURM caveat:** the array job id isn't known until `sbatch` returns. Snapshot to a temp
   ref first, then rename to `refs/sweeps/<jobid>` once we have the id.
3. **Reproduce later** (shown in the dashboard, copy-paste):
   - inspect the diff: `git diff <base_commit> <repro_snapshot>`
   - check out into an isolated worktree (doesn't disturb current work):
     `git worktree add /tmp/repro_<id> <repro_snapshot>`
   - or apply onto a clean base: `git stash apply <repro_snapshot>`
4. **Dashboard:** the 5a panel gains "code snapshot `refs/sweeps/<id>` @ `<sha>`", a
   **`Show diff vs base`** expander (renders `git diff base..snapshot`), and the worktree
   command.

**Why this satisfies the constraint:** `git stash create` never moves HEAD, never stages,
never modifies the working tree or the stash list; the snapshot lives only under
`refs/sweeps/*`, so the dev branch is byte-for-byte unaffected and nothing leaks into normal
history or pushes. The commit is kept alive (not GC'd) precisely because the ref points at it.

**Implementation / files**
- New `reporting/repro.py`: `snapshot_worktree(sweep_id) -> {sha, ref, base, branch, dirty}`,
  `rename_ref`, `diff(base, sha)`, `worktree_cmd(sha)`. Pure `subprocess` git plumbing.
- `sweep_train_al.py`: call `snapshot_worktree` inside `launch_local_bg` and
  `submit_sbatch_jobs`; thread the result into the launch info / registry entry.
- `registry.py`: carry the new repro fields through `make_entry`/`upsert`.
- `dashboard_next.py`: extend the 5a panel with snapshot + diff + worktree.

**Effort:** H. **Risk:** medium–high — git plumbing edge cases (detached HEAD, submodules,
empty tree, untracked files: `git stash create` ignores untracked unless we also snapshot
them — may need `git add -A` into a temp index, see open Q), and the SLURM id-after-submit
dance. Worth a small spike on `git stash create` + `update-ref` first.
**Open Q:** include **untracked** files in the snapshot? (`stash create` skips them; a temp-index
`write-tree` captures them. Probably yes for true reproducibility.) Auto-snapshot every launch,
or an opt-in checkbox? (Proposal: on by default, cheap; checkbox to skip.)

---

## Suggested sequencing

| # | Feature | Effort | Depends on | Note |
|---|---------|--------|-----------|------|
| 0 | Sweep description cards | S | — | registry `title`/`context` + live auto-facts; the "make the mess legible" win |
| 1 | Deprecate/flag sweeps | S | 0 | shares the registry write-back path; immediate cleanup win |
| 2 | X/Y axis selection | S–M | — | one shared scanner change (widen aggregator) |
| 5a | "Reproduce this curve" panel | S | 0 | free; plugs into Feature 0's "About this curve" panel |
| 3 | Filter→group-by facets | M | 2 | generalise NPC panel into `_facet_grid` |
| 4 | Shareable URL state | M–H | 1–3 | serialise the now-stable control set |
| 5b | Pre-sweep code snapshot | H | 5a | git `stash create` + `refs/sweeps/*`; spike first |

Recommended order: **0 → 1 → 2 → 5a → 3 → 4 → 5b** (front-load the cheap high-value wins; do
the git spike for 5b last). Features 0+1 are best done together — they touch the same registry
write-back and Ledger card. Each is an independent PR-sized change against `dashboard_next.py`.

## For the implementer (fresh session — start here)

Everything below was verified on disk 2026-06-11; line numbers may drift, grep to confirm.

**Key files & anchors**
- `reporting/dashboard_next.py` — the live dashboard (replaces `dashboard.py`). Structure:
  module-level helpers + `_render_ledger / _render_comparison / _render_additional /
  _render_npc / _render_details / _render_export / _render_launch` functions, then a
  3-tab + sub-tab layout block at the **bottom** wires them up. Graphs work lives in
  `_render_comparison` (Features 2/3/4) and `_render_ledger` (Features 0/1).
- `reporting/scanner.py` — `load_all_results()` (~L96, the per-run df + `p_*` columns),
  `aggregate_config_episodes()` (~L194, **widen this for Feature 2**), `get_all_aggregated()`.
- `reporting/registry.py` — `make_entry()` (~L154), `upsert()` (~L115, preserves user fields),
  `set_category()` (~L144); add `title/context/flags` here and mirror the preserve logic.
  Backed by `logs/sweep/sweep_registry.json`.
- `reporting/backfill_registry.py` — must keep preserving any new user fields (note rule).
- `batch_utils/sweep_train_al.py` — `launch_local_bg()` (~L269), `submit_sbatch_jobs()` (~L322):
  the two launch paths to hook for Feature 5b's snapshot.
- `reporting/html_export.py` — `generate_report()` (used by the Export sub-tab).

**Per-run df columns:** `exp_dir, result_file, status, created_at, finished_at, sweep_id,
sweep_name, slurm_task_id, commit_hash, command, config_key, config_label,
total_episodes_planned/done, final_test_acc, episodes, error, p_*`.
**Per-episode fields:** `episode, test_acc, val_acc, train_acc, labeled_size, duration_s`.

**Smoke test after every change** (headless, no browser):
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("reporting/dashboard_next.py", default_timeout=120).run()
assert len(at.exception) == 0, at.exception
```
Run it inside the repo with `source venv/bin/activate` first. The `noise_removal_aum` branch
is the working branch.

**Decisions still pending (resolve with alon before/while building — see below):**
budget-axis formula (F2), flag vocabulary (F1), snapshot scope + auto-vs-opt-in (F5b),
description granularity sweep-vs-curve (F0), and whether to delete `dashboard.py` + update the
`al-dashboard` skill once F0–F2 land.

## Cross-cutting open questions
1. **Budget axis accounting** — exact cumulative oracle-cost formula (does label noise inflate
   by 1/τ?). Blocks the "budget" X-axis option in Feature 2.
2. **Flag vocabulary** — fixed `{bug, deprecated}` vs free tags (Feature 1).
3. **Snapshot scope** — include untracked files? auto vs opt-in? (Feature 5b).
4. **Retire `dashboard.py`** — once Features 1–2 land in `next`, delete the old dashboard and
   update the `al-dashboard` skill (currently still documents the old 6-tab UI).

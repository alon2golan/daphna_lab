"""
Sweep registry: a single JSON ledger of every sweep/job you've run, of any
kind (train_al experiments, ν'-caching jobs, other-repo jobs, misc tasks).

This is the data source for the new dashboard's **Sweeps** tab. Unlike the
result.json scanner (which only sees train_al runs), the registry can hold ANY
job — seeded from history by reporting/backfill_registry.py and appended to as
you launch new sweeps.

Schema — logs/sweep/sweep_registry.json is a list of entries:
    {
      "sweep_id":   "30726412",                    # SLURM array job id, or local_/manual_ id
      "sweep_name": "fig55_daphna_20260603_104536",
      "kind":       "train_al",                    # train_al | nu_caching | other_repo | misc
      "category":   "fig55",                        # finer purpose tag (see CATEGORY_RULES)
      "command":    "...",                          # representative command / script
      "created_at": "2026-06-03T10:45:38",          # ISO; "" if unknown
      "source":     "result_json",                  # result_json | sacct | logs | manual
      "note":       "",                             # one-line free text, never wiped by backfill
      "title":      "",                             # curated short title (Feature 0)
      "context":    "",                             # curated multi-line markdown story (Feature 0)
      "flags":      []                              # e.g. ["deprecated"], ["bug"] (Feature 1)
    }

The (kind, category) pair is auto-derived by ``categorize()`` but is freely
overridable — the dashboard writes user edits straight back here, and the
backfill preserves any entry whose ``source == "manual"`` plus the curated
user fields (``note``, ``title``, ``context``, ``flags``, ``category``) of every
existing entry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = REPO_ROOT / "logs" / "sweep" / "sweep_registry.json"

# ── Categorisation ─────────────────────────────────────────────────────────────
#
# Ordered; first match wins. Each rule matches against "<sweep_name> <command>".
# Edit freely — this is the single source of truth for auto-categories.
#
# `category` is the finer purpose tag used to scope the Graphs tab; `kind` is the
# coarse bucket the Sweeps ledger groups by when a job isn't a train_al run.

KINDS = ["train_al", "nu_caching", "other_repo", "misc"]

# Flag vocabulary (Feature 1). Fixed for now — the list field leaves room to add
# free-form tags later. Flags in HIDDEN_FLAGS hide a sweep from the Graphs/Ledger
# pickers by default (one toggle reveals them).
FLAG_VOCAB = ["deprecated", "bug"]
HIDDEN_FLAGS = {"deprecated", "bug"}

# Curated, user-owned fields the backfill must never clobber.
_USER_FIELDS = ("category", "note", "title", "context", "flags")

# Purpose tags for train_al runs (the Graphs tab scopes on these).
CATEGORY_RULES = [
    # Tomer's Fig-5.5 embedding graphs (sweep names still carry the fig55 token).
    ("tomer_embedding_graphs", re.compile(r"fig55")),
    # NPC / probcover / random × {train-on-clean on,off} × noise × seeds, all
    # sharing the AUM noise filter. Matched on the "AUMFILTER" sweep-name token.
    ("aum_filter",      re.compile(r"aumfilter|aum_filter", re.IGNORECASE)),
    ("oracle_vs_noisy", re.compile(r"probcover_random|oracle_vs_noisy")),
    # The 3000-oracle-cost-cap family (probcover/random × noisy-τ AND oracle arms,
    # ν'-cache decoupled from budget). Matched on the shared "3000cap" sweep-name
    # token so both the noisy and oracle trainings land in one category. Placed
    # before nu_estimation because these commands carry "--nu_prime"/"nucache..."
    # tokens that rule could otherwise grab.
    ("train_noise_nu_prime", re.compile(r"3000cap")),
    # noise_mechanism before nu_estimation: "bypass_noise_mechanism" sweeps embed
    # a "_nu0.3" in their command, which the nu_estimation rule would match first.
    ("noise_mechanism", re.compile(r"bypass_noise|noise_mechanism")),
    ("nu_estimation",   re.compile(r"oneshot|seeddecoupled|nu_sweep|_nu[0-9.]")),
]
KNOWN_CATEGORIES = (
    [c for c, _ in CATEGORY_RULES]
    + ["uncategorized", "nu_caching", "other_repo", "misc"]
)

# Categories whose runs are "track-progress" by nature: their accuracy-vs-episode
# curves belong in the dashboard Comparison tab. Used as the fallback for runs
# that predate the per-run --track_progress flag (the recorded flag always wins).
# Everything else (fig55, nu_caching, npc-repro diagnostics, …) is a figure
# ingredient and surfaces under Additional data instead.
TRACK_PROGRESS_CATEGORIES = {"train_noise_nu_prime", "aum_filter"}

_NU_CACHE_RX = re.compile(r"nu_prime|nu_optcompare|sweep_nu", re.IGNORECASE)
_TRAIN_AL_RX = re.compile(r"train_al\.py")

# Category rules for results/ ARTIFACTS (figures), matched against the file path.
# Separate from CATEGORY_RULES because figure filenames differ from sweep names
# (e.g. the run rule "_nu[0-9.]" would false-match "tau_vs_nu.png"). First match wins.
ARTIFACT_CATEGORY_RULES = [
    ("tomer_embedding_graphs", re.compile(r"tomer_embedding|fig55", re.IGNORECASE)),
    ("train_noise_nu_prime", re.compile(r"3000cap", re.IGNORECASE)),
    ("aum_filter",           re.compile(r"aumfilter|aum_filter", re.IGNORECASE)),
    ("oracle_vs_noisy",      re.compile(r"repro_4_15|oracle_vs_noisy|probcover_random", re.IGNORECASE)),
    ("npc_repro",            re.compile(r"netta|npc", re.IGNORECASE)),
    ("nu_caching",           re.compile(r"nu_prime|nu_optcompare|nu_vs_tau|tau_vs_nu|heatmap", re.IGNORECASE)),
]


def categorize_artifact(path: str) -> str:
    """Best-effort category for a results/ figure, by matching its path against
    ARTIFACT_CATEGORY_RULES (first match wins); 'misc' if nothing matches."""
    s = path or ""
    for cat, rx in ARTIFACT_CATEGORY_RULES:
        if rx.search(s):
            return cat
    return "misc"


def categorize(sweep_name: str, command: str = "") -> dict:
    """Return {"kind": ..., "category": ...} derived from name + command."""
    hay = f"{sweep_name or ''} {command or ''}"

    if _TRAIN_AL_RX.search(hay):
        kind = "train_al"
        category = "uncategorized"
        for name, rx in CATEGORY_RULES:
            if rx.search(hay):
                category = name
                break
        return {"kind": kind, "category": category}

    if _NU_CACHE_RX.search(hay):
        return {"kind": "nu_caching", "category": "nu_caching"}

    # Unknown script we can still name → other_repo if it looks like a real job,
    # else misc. We can't reliably tell, so default everything else to misc.
    return {"kind": "misc", "category": "misc"}


# ── Load / save ─────────────────────────────────────────────────────────────────


def load_registry() -> list[dict]:
    if REGISTRY_FILE.exists():
        try:
            data = json.loads(REGISTRY_FILE.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_registry(entries: list[dict]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Stable order: newest first by created_at, unknown dates last.
    entries = sorted(
        entries,
        key=lambda e: (e.get("created_at") or "", e.get("sweep_id") or ""),
        reverse=True,
    )
    REGISTRY_FILE.write_text(json.dumps(entries, indent=2))


def _index(entries: list[dict]) -> dict[str, dict]:
    return {e["sweep_id"]: e for e in entries if e.get("sweep_id")}


def upsert(entry: dict, *, preserve_user_fields: bool = True) -> None:
    """
    Insert or merge a single entry, keyed on sweep_id.

    When merging an existing entry, user-owned fields (``category``, ``note``,
    ``title``, ``context``, ``flags`` — see ``_USER_FIELDS`` — and any entry whose
    ``source == "manual"``) are preserved by default so a re-run of the backfill
    never clobbers hand edits.
    """
    entries = load_registry()
    idx = _index(entries)
    sid = entry.get("sweep_id")
    if not sid:
        return
    if sid in idx and preserve_user_fields:
        existing = idx[sid]
        if existing.get("source") == "manual":
            return  # fully user-owned; leave untouched
        merged = {**entry}
        for k in _USER_FIELDS:
            if existing.get(k):
                merged[k] = existing[k]
        idx[sid].update(merged)
    elif sid in idx:
        idx[sid].update(entry)
    else:
        entries.append(entry)
    save_registry(entries)


def _set_fields(sweep_id: str, **fields) -> bool:
    """Set arbitrary fields on one existing entry. Returns True if found."""
    entries = load_registry()
    for e in entries:
        if e.get("sweep_id") == str(sweep_id):
            e.update(fields)
            save_registry(entries)
            return True
    return False


def set_category(sweep_id: str, category: str) -> None:
    """Override the category of one sweep (called by the dashboard dropdown)."""
    _set_fields(sweep_id, category=category)


def set_description(sweep_id: str, title: str = None, context: str = None) -> bool:
    """Set the curated title and/or context (Feature 0). None = leave unchanged."""
    fields = {}
    if title is not None:
        fields["title"] = title
    if context is not None:
        fields["context"] = context
    return _set_fields(sweep_id, **fields) if fields else False


def set_flags(sweep_id: str, flags: list, note: str = None) -> bool:
    """
    Replace the flags list of one sweep (Feature 1), optionally setting the note.
    Pass flags=[] to clear ("restore"). Unknown flags are kept as-is so callers
    can add free-form tags later.
    """
    fields = {"flags": list(dict.fromkeys(flags))}
    if note is not None:
        fields["note"] = note
    return _set_fields(sweep_id, **fields)


def make_entry(
    sweep_id: str,
    sweep_name: str,
    command: str = "",
    created_at: str = "",
    source: str = "manual",
    note: str = "",
    title: str = "",
    context: str = "",
    flags: list | None = None,
) -> dict:
    cat = categorize(sweep_name, command)
    return {
        "sweep_id": str(sweep_id),
        "sweep_name": sweep_name or str(sweep_id),
        "kind": cat["kind"],
        "category": cat["category"],
        "command": command,
        "created_at": created_at,
        "source": source,
        "note": note,
        "title": title,
        "context": context,
        "flags": list(flags) if flags else [],
    }


# ── CLI: hand-add an entry ───────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Manually add a sweep entry, or edit fields of an existing one.\n"
        "Examples:\n"
        "  # add/replace an entry\n"
        "  python reporting/registry.py --id 30811243 --name my_sweep --note '...'\n"
        "  # edit just the curated story of an existing sweep\n"
        "  python reporting/registry.py --id 30797616 --edit \\\n"
        "      --title \"ν' cache dense sweep\" --context 'PC+random, fixed cache'\n"
        "  # deprecate / flag an existing sweep, with a reason\n"
        "  python reporting/registry.py --id 30811243 --edit --flag deprecated --note 'buggy noise seed'\n"
        "  # clear all flags (restore)\n"
        "  python reporting/registry.py --id 30811243 --edit --unflag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--id", required=True, help="sweep_id (SLURM array job id or any unique id)")
    p.add_argument("--name", help="human-readable sweep name (required unless --edit)")
    p.add_argument("--command", default="", help="representative command / script")
    p.add_argument("--created-at", default="", help="ISO timestamp (optional)")
    p.add_argument("--category", default=None, help="override auto-category")
    p.add_argument("--note", default=None, help="one-line free-text note")
    p.add_argument("--title", default=None, help="curated short title (Feature 0)")
    p.add_argument("--context", default=None, help="curated multi-line story (Feature 0)")
    p.add_argument("--flag", action="append", default=None, metavar="FLAG",
                   help=f"add a flag (repeatable); known: {FLAG_VOCAB}")
    p.add_argument("--unflag", action="store_true", help="clear all flags (restore)")
    p.add_argument("--edit", action="store_true",
                   help="edit fields of an EXISTING entry instead of adding a new one")
    a = p.parse_args()

    if a.edit:
        if a.unflag:
            set_flags(a.id, [], note=a.note)
        elif a.flag:
            entry = next((e for e in load_registry() if e.get("sweep_id") == str(a.id)), {})
            merged = list(dict.fromkeys((entry.get("flags") or []) + a.flag))
            set_flags(a.id, merged, note=a.note)
        if a.title is not None or a.context is not None:
            set_description(a.id, title=a.title, context=a.context)
        if a.category:
            set_category(a.id, a.category)
        if a.note is not None and not (a.unflag or a.flag):
            _set_fields(a.id, note=a.note)
        print(f"Edited entry `{a.id}` → {REGISTRY_FILE}")
    else:
        if not a.name:
            p.error("--name is required when adding a new entry (use --edit to modify an existing one)")
        e = make_entry(a.id, a.name, a.command, a.created_at, source="manual",
                       note=a.note or "", title=a.title or "", context=a.context or "",
                       flags=a.flag)
        if a.category:
            e["category"] = a.category
        upsert(e, preserve_user_fields=False)
        print(f"Registry now has {len(load_registry())} entries → {REGISTRY_FILE}")

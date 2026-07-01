import pandas as pd
from reporting import scanner, registry as reg

df = scanner.load_all_results()
sgd = df[df["sweep_name"].astype(str).str.contains("train_al_3000cap_sgd")].copy()

# replicate dashboard track_progress derivation
def cat_for(row):
    return reg.categorize(row.get("sweep_name", ""), row.get("command", ""))["category"]
def tp_for(row):
    tp = row.get("p_track_progress")
    if tp is None or (isinstance(tp, float) and pd.isna(tp)):
        return cat_for(row) in reg.TRACK_PROGRESS_CATEGORIES
    return bool(tp)

sgd["category"] = sgd.apply(cat_for, axis=1)
sgd["track_progress"] = sgd.apply(tp_for, axis=1)

print("category(s):", sgd["category"].unique().tolist())
print("p_track_progress values:", sgd["p_track_progress"].unique().tolist())
print("track_progress True count:", int(sgd["track_progress"].sum()), "/", len(sgd))
print("\nrandom+sgd rows visible in DEFAULT Comparison (track_progress only):",
      int(((sgd.p_al_method == 'random') & sgd.track_progress).sum()))
print("random+sgd rows visible WITH 'show non-tracked' ticked:",
      int((sgd.p_al_method == 'random').sum()))

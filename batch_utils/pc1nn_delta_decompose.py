import numpy as np
from pc1nn_delta_replay import run

def stat(accs):
    a=np.array(accs); return f"{a.mean():5.2f} ± {a.std():.2f}  (seeds: {', '.join(f'{x:.2f}' for x in a)})"

print("="*70)
print("BASELINE (ours): itai, delta=0.75, argmax")
print("  ", stat([run("itai",s,0.75,"argmax")[0] for s in (1,2,3)]))

print("\n[1] SELECTION MODE  (itai, delta=0.75)")
print("  argmax (ours) :", stat([run("itai",s,0.75,"argmax")[0] for s in (1,2,3)]))
print("  rand_top5(pap):", stat([run("itai",s,0.75,"rand_top5")[0] for s in (1,2,3)]))

print("\n[2] DELTA SWEEP  (itai, argmax, mean over seeds 1-3)")
for d in (0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90):
    accs=[run("itai",s,d,"argmax")[0] for s in (1,2,3)]
    print(f"  delta={d:.2f}:", stat(accs))

print("\n[3] EMBEDDING  (delta=0.75, argmax)")
print("  itai (used)   :", stat([run("itai",s,0.75,"argmax")[0] for s in (1,2,3)]))
print("  alon          :", stat([run("alon",s,0.75,"argmax")[0] for s in (1,2,3)]))

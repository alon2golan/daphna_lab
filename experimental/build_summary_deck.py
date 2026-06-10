"""
build_summary_deck.py
---------------------
Generate a self-contained HTML presentation summarizing the NoAA
reproduction + debugging work (Tomer Fig 5.5, the oracle-vs-noisy bug, the
seed-coupling noise bug, the Adam-vs-SGD ν′ difference, and the 4_15
reproduction). All figures are base64-embedded so the file is portable.

Re-run any time (e.g. after the 4_15 repro sweep lands) to refresh:
    venv/bin/python experimental/build_summary_deck.py
Output: results/noaa_summary_deck.html
"""
from __future__ import annotations

import base64
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "noaa_summary_deck.html"


def img(path: Path, alt: str = "") -> str:
    if not path.exists():
        return (f'<div class="missing">[figure not yet available: {path.name}]</div>')
    b64 = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f'<img alt="{alt}" src="data:image/{mime};base64,{b64}"/>'


def read_b300_table() -> str:
    p = REPO / "results" / "cifar100" / "repro_4_15" / "B300_compare_gap_table.csv"
    if not p.exists():
        return '<div class="missing">[B=300 comparison not in yet]</div>'
    rows = list(csv.DictReader(open(p)))
    head = ("<tr><th>strategy</th><th>τ</th><th>oracle</th>"
            "<th>noisy<br>B100-ν′</th><th>gap</th>"
            "<th>noisy<br>B300-ν′</th><th>gap</th><th>Δacc</th></tr>")
    body = ""
    for r in rows:
        def cls(v):
            try:
                return "neg" if float(v) < 0 else "pos"
            except (TypeError, ValueError):
                return ""
        body += (f"<tr><td>{r['strategy']}</td><td>{float(r['tau']):.1f}</td>"
                 f"<td>{r['oracle']}</td>"
                 f"<td>{r['noisy_B100']}</td><td class='{cls(r['gap_B100'])}'>{r['gap_B100']}</td>"
                 f"<td>{r['noisy_B300']}</td><td class='{cls(r['gap_B300'])}'>{r['gap_B300']}</td>"
                 f"<td class='pos'>+{r['delta']}</td></tr>")
    return f"<table class='data'>{head}{body}</table>"


def read_gap_table() -> str:
    p = REPO / "results" / "cifar100" / "repro_4_15" / "gap_table.csv"
    if not p.exists():
        return '<div class="missing">[after-data not in yet — gap table will appear once job 30742908 finishes]</div>'
    rows = list(csv.DictReader(open(p)))
    head = ("<tr><th>strategy</th><th>τ</th>"
            "<th>oracle<br>before</th><th>noisy<br>before</th><th>gap<br>before</th>"
            "<th>oracle<br>after</th><th>noisy<br>after</th><th>gap<br>after</th></tr>")
    body = ""
    for r in rows:
        gb = r["gap_before"]; ga = r["gap_after"]
        def cls(v):
            try:
                return "neg" if float(v) < 0 else "pos"
            except (TypeError, ValueError):
                return ""
        body += (f"<tr><td>{r['strategy']}</td><td>{float(r['tau']):.1f}</td>"
                 f"<td>{r['oracle_before']}</td><td>{r['noisy_before']}</td>"
                 f"<td class='{cls(gb)}'>{gb}</td>"
                 f"<td>{r['oracle_after']}</td><td>{r['noisy_after']}</td>"
                 f"<td class='{cls(ga)}'>{ga}</td></tr>")
    return f"<table class='data'>{head}{body}</table>"


# --- slide content -----------------------------------------------------------
FIGS = REPO / "results"
DIAG = REPO / "diagnostics"
C100 = REPO / "results" / "cifar100"

SLIDES = []

SLIDES.append(("title", """
  <h1>Active Learning with Two Annotators — NoAA</h1>
  <h2>Reproducing Tomer Yaacoby's results in <code>daphna_lab</code>, and the three bugs we found</h2>
  <p class="sub">CIFAR-100 · linear classifier on SimCLR features · ProbCover / Random sampling<br>
  Alon Golan · Weinshall lab · summary as of 2026-06-04</p>
  <div class="tldr">
    <b>TL;DR</b> — We reproduced Tomer's Fig 5.5 (NoAA tracks the ground-truth budget split).
    Along the way we found and fixed <b>three</b> issues that had made the noisy annotator look
    artificially worse than the oracle:
    <ol>
      <li>a <b>seed-coupling RNG bug</b> that inflated realized label noise (45% at a nominal 30%, seed 1);</li>
      <li>a <b>budget / padding</b> artifact at B=3000 in the original parquet; and</li>
      <li>an <b>optimizer mismatch</b> (Adam vs SGD) in the ν′ estimator.</li>
    </ol>
    After the fixes, the noisy annotator at the equilibrium ν′ <b>matches and at low noise surpasses</b> the oracle — exactly as NoAA predicts.
  </div>
"""))

SLIDES.append(("Background", """
  <h2>The question</h2>
  <p>NoAA (Noise-Aware Annotation): given a <b>perfect oracle</b> annotator and a <b>cheap noisy</b> one,
  how should we split a fixed labeling budget? Tomer's <b>Algorithm 1</b> estimates
  <b>ν′(B, τ)</b> — the noise rate at which "all-noisy on B/τ cheap labels" gives the
  <i>same</i> downstream accuracy as "all-oracle on B clean labels".</p>
  <ul>
    <li><b>τ</b> = cost ratio (cheap label costs τ of an oracle label) → noisy annotator yields B/τ labels for the same budget.</li>
    <li>At ν &lt; ν′(τ) the noisy annotator should <b>beat</b> the oracle; at ν &gt; ν′ it should lose.</li>
  </ul>
  <p><b>Our goal:</b> reproduce Tomer's figures natively inside <code>daphna_lab</code> (a newer code layout),
  and validate the AL pipeline + the ν′ estimator against his thesis data.</p>
"""))

SLIDES.append(("Fig 5.5 reproduced", f"""
  <h2>✅ Fig 5.5 reproduced — NoAA tracks Ground Truth</h2>
  {img(FIGS / 'fig55_daphna.png', 'fig55 reproduction')}
  <p>Six strategies × three τ panels, CIFAR-100, ProbCover, post-fix. In every panel the
  <b>NoAA</b> curve (brown) tracks <b>Ground Truth</b> (purple); All-Oracle (blue) is lowest at small budgets.
  Qualitatively identical to the original <code>oracle-teacher-proj</code> figure. The native daphna_lab
  reproduction succeeds.</p>
"""))

SLIDES.append(("The reported bug", f"""
  <h2>⚠️ The bug Tomer's graph showed: noisy &lt; oracle at small τ</h2>
  <p>The original sweep <code>probcover_random__RESNET18__seed1to3__20260415_153802</code>
  (B=100, 11 AL episodes, seeds 1–3) showed the noisy annotator <b>degraded vs the oracle for every τ</b>,
  contradicting NoAA — which says at the equilibrium ν′(τ) it should ≈ oracle.</p>
  <table class="data">
    <tr><th>strategy</th><th>τ=1.0 oracle</th><th>τ=0.8</th><th>τ=0.5</th><th>τ=0.2</th></tr>
    <tr><td>probcover</td><td>47.29</td><td>45.68</td><td>44.13</td><td>41.91</td></tr>
    <tr><td>random</td><td>43.53</td><td>37.39</td><td>36.08</td><td>35.45</td></tr>
  </table>
  <p class="cap">Seed-averaged ep-10 test acc, <b>original (buggy)</b> runs. Oracle dominates everywhere — strongest for <code>random</code> (43.5 vs ~36).
  This "degradation" is the artifact we set out to explain.</p>
"""))

SLIDES.append(("Root cause #1", f"""
  <h2>🐞 Bug #1 — seed-coupling inflated the realized noise</h2>
  <p>At seed 1 the annotator trained on <b>44.9% corrupted labels at a nominal 30%</b> (14.8σ above
  chance — deterministic, not sampling). Cause: the noise RNG was seeded with
  <code>cfg.RNG_SEED</code>, the <i>same</i> seed that shuffles the train/val split. Since
  numpy's <code>choice(N,k,replace=False)</code> is the same Fisher–Yates permutation, the
  noise mask landed entirely in the train pool <i>and</i> ProbCover over-picked the noisy block.</p>
  <div class="cols">
    <div>
      <p><b>The fix</b> (<code>annotator.py</code>): a dedicated noise RNG
      <code>RandomState(seed + NOISE_SEED_OFFSET)</code>, decoupled from <code>RNG_SEED</code>.</p>
      <p><b>Verification</b> (single-shot B=1000, τ=0.5, ν=0.3, 5 seeds):</p>
      <table class="data">
        <tr><th></th><th>realized noise</th><th>test acc</th></tr>
        <tr><td>before (native)</td><td class="neg">0.449</td><td>42.20</td></tr>
        <tr><td>after (fixed)</td><td class="pos">~0.30</td><td>44.14 ± 0.6</td></tr>
        <tr><td>Tomer parquet</td><td>0.293</td><td>44.55 ± 0.57</td></tr>
      </table>
      <p>Gap closed from ~2.4 pt → ~0.4 pt.</p>
    </div>
    <div>{img(DIAG / 'nu_overlay_B1000_tau0p5.png', 'nu overlay')}
    <p class="cap">Pre-fix daphna-native noisy curve (red) sat below Tomer (blue); the gap grew with ν —
    consistent with the harsher realized noise.</p></div>
  </div>
"""))

SLIDES.append(("The recheck", """
  <h2>🔁 The recheck: noisy actually <i>surpasses</i> the oracle</h2>
  <p>Two independent confirmations that, once noise is correct, NoAA holds:</p>
  <ul>
    <li><b>Label bypass.</b> Feeding Tomer's <i>actual</i> noisy labels into the daphna_lab pipeline
      recovers <b>44.74</b> (5-rep) ≈ Tomer 44.55 — vs 42.20 with the buggy native noise. The training
      pipeline was never the problem; noise generation was.</li>
    <li><b>Post-fix 4_15 re-run.</b> Per-episode, at <b>ep 0 the noisy annotator BEATS the oracle</b>
      (τ=0.5: −3.0 pt; τ=0.2: −1.5 pt). The residual late-episode gap is a <b>budget</b> effect:
      ν′ is looked up once at B=100 but the multi-episode run accumulates to ~2000 noisy samples, where
      ν′ is off-equilibrium.</li>
  </ul>
  <p>So the original "noisy is worse" was an artifact (inflated noise + fixed-budget ν′), not a failure of NoAA.</p>
"""))

SLIDES.append(("Root cause #2", f"""
  <h2>⚙️ Bug #2 — the ν′ estimator's optimizer (Adam vs SGD)</h2>
  <p>daphna_lab's estimator hardcoded <b>Adam</b>; Tomer's port used <b>SGD + weight-decay + LR schedule</b>.
  Adam overfits noisy labels → a <i>lower</i> equilibrium ν′, biggest where ν′ is large (small τ). We added a
  <code>--optimizer</code> switch and re-estimated ν′ for ProbCover at all budgets, both optimizers.</p>
  {img(FIGS / 'fig55_daphna_noaa_optcompare.png', 'noaa adam vs sgd')}
  <p class="cap">NoAA-SGD (brown) vs NoAA-Adam (magenta dashed). For B ≤ 1000 both ≈ Ground Truth. At <b>B=3000</b> they
  diverge: NoAA-Adam tracks GT better; NoAA-SGD's ν′ collapses (its flat-LR SGD <i>underfits</i> the large 10k–30k noisy sets).</p>
"""))

SLIDES.append(("B=3000 caveat", """
  <h2>🧪 Why the estimator is unreliable at B=3000</h2>
  <p>We chased the B=3000 NoAA endpoint (which collapses onto the oracle) and found <b>two</b> causes:</p>
  <ol>
    <li><b>Padding (real bug, not the root):</b> the original parquet could only assemble ~14.5k unique
      samples for the remapped budget=30000 and <b>padded by repeating one image ~15,400×</b> (51% of the set).
      Fixed by dedupe + top-up. On clean lsets ν′ still falls short → padding alone doesn't explain it.</li>
    <li><b>Proxy saturation (dominant):</b> the estimator scores against <b>KMeans cluster pseudo-labels</b>, not
      real classes. At B=3000 the oracle already saturates the easy cluster task (~0.78); the noisy model can at
      best tie it, so ν′ collapses 2–3× below the true (real-label) crossover for both optimizers.</li>
  </ol>
  <p><b>Net:</b> the ν′ estimator is out-of-regime at B=3000. It's trustworthy for B ≤ 1000 (where NoAA matches GT).
  Options: a harder proxy (more clusters), flag/omit the B=3000 point, or use the clean-Adam values as a stopgap.</p>
"""))

SLIDES.append(("4_15 reproduction", f"""
  <h2>📈 The 4_15 figure reproduced — both strategies, post-fix, NoAA-Adam ν′</h2>
  {img(C100 / 'repro_4_15' / 'repro_4_15_before_after.png', '4_15 before/after')}
  {read_gap_table()}
  <p class="cap"><b>gap = oracle − noisy</b> at ep-10 (seed-avg). <span class="neg">Negative/green-shrinking</span> = noisy caught up.
  Before vs after the seed-decoupling fix, using the same Adam ν′ cache.</p>
"""))

SLIDES.append(("Budget-matched ν′", f"""
  <h2>🎯 The residual gap is a budget effect — B=300 ν′ closes it</h2>
  <p>ν′ is budget-dependent and <i>decreases</i> with B. The run is looked up once at B=100, but its
  labeled set accumulates to ~1100–2200 by ep10, so the B=100 ν′ <b>over-noises</b> the later episodes.
  Re-estimating ν′ at <b>B=300</b> (Adam, same estimator; probcover exact-lset, random fallback) and feeding
  it via <code>--nu_prime</code> (training still at B=100) raises every noisy curve toward the oracle:</p>
  {img(C100 / 'repro_4_15' / 'repro_4_15_B300_compare.png', 'B100 vs B300 nu prime')}
  {read_b300_table()}
  <p class="cap">ep10 seed-avg. The lower B=300 ν′ shrinks every gap; at τ=0.2 the noisy annotator <b>matches</b>
  the oracle (probcover gap 2.41→0.16) or <b>beats</b> it (random gap 2.96→<span class="neg">−0.21</span>) — confirming the
  gap was ν′/budget mis-calibration, not a pipeline failure. B=300 ν′: probcover {{0.439, 0.238, 0.077}}, random {{0.569, 0.382, 0.129}}.</p>
"""))

SLIDES.append(("ν′ for random + status", """
  <h2>📦 Do we have ν′ (Adam + SGD) for <code>random</code>?</h2>
  <table class="data">
    <tr><th>estimator</th><th>probcover</th><th>random</th></tr>
    <tr><td><b>Adam</b> ν′</td><td>✅ yes (cache + optcompare)</td><td>✅ <b>yes</b> — <code>nu_prime_cache.json</code>, B∈{100,200,500}, all τ</td></tr>
    <tr><td><b>SGD</b> ν′</td><td>✅ yes (optcompare, B∈{100,300,1000,3000})</td><td>❌ <b>no</b> — optcompare was probcover-only</td></tr>
  </table>
  <p>The 4_15 reproduction above uses the <b>Adam</b> ν′ cache for both strategies (the values the original sweep used).
  A NoAA-<b>SGD</b> comparison for <code>random</code> would need a fresh estimator sweep with <code>--optimizer sgd</code>.</p>
  <h3>Status</h3>
  <ul>
    <li>✅ Fig 5.5 reproduced natively in daphna_lab (NoAA tracks GT).</li>
    <li>✅ Three issues found & fixed: seed-coupling noise, B=3000 padding, Adam-vs-SGD ν′.</li>
    <li>✅ 4_15 sweep re-run for probcover <b>and</b> random with the fixed annotator (job 30742908).</li>
    <li>🔜 Optional: SGD ν′ for random; harder KMeans proxy at B=3000.</li>
  </ul>
"""))


def render() -> str:
    nav = "".join(f'<button onclick="go({i})">{i+1}</button>' for i in range(len(SLIDES)))
    slides_html = ""
    for i, (tag, body) in enumerate(SLIDES):
        cls = "slide title-slide" if tag == "title" else "slide"
        slides_html += f'<section class="{cls}" data-i="{i}"><div class="chip">{tag}</div>{body}</section>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoAA reproduction — summary</title>
<style>
  :root {{ --bg:#0f1221; --card:#1a1f38; --ink:#e8ecff; --mut:#9aa3c7; --acc:#7aa2ff;
           --pos:#34c759; --neg:#ff6b6b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .deck {{ max-width:1080px; margin:0 auto; padding:24px 20px 90px; }}
  .slide {{ background:var(--card); border:1px solid #2a3157; border-radius:16px;
            padding:30px 34px; margin:22px 0; scroll-margin-top:20px;
            box-shadow:0 8px 30px rgba(0,0,0,.35); }}
  .title-slide {{ text-align:center; padding:54px 40px; }}
  h1 {{ font-size:30px; margin:.1em 0 .25em; }}
  h2 {{ font-size:23px; margin:.1em 0 .5em; color:var(--acc); }}
  h3 {{ font-size:18px; margin:1em 0 .3em; color:var(--ink); }}
  .sub {{ color:var(--mut); }}
  .chip {{ display:inline-block; font-size:12px; letter-spacing:.08em; text-transform:uppercase;
           color:var(--mut); border:1px solid #2a3157; border-radius:999px; padding:3px 12px; margin-bottom:14px; }}
  img {{ max-width:100%; border-radius:10px; background:#fff; padding:6px; margin:10px 0; }}
  code {{ background:#0c0f1f; padding:1px 6px; border-radius:6px; color:#cdd6ff; font-size:.92em; }}
  ul,ol {{ margin:.4em 0 .6em; padding-left:1.3em; }} li {{ margin:.28em 0; }}
  table.data {{ border-collapse:collapse; margin:14px 0; font-size:14px; width:100%; }}
  table.data th, table.data td {{ border:1px solid #2a3157; padding:7px 11px; text-align:center; }}
  table.data th {{ background:#0c0f1f; color:var(--mut); font-weight:600; }}
  td.neg {{ color:var(--neg); font-weight:700; }} td.pos {{ color:var(--pos); font-weight:700; }}
  .cap {{ color:var(--mut); font-size:13.5px; }}
  .tldr {{ text-align:left; background:#0c0f1f; border:1px solid #2a3157; border-radius:12px;
           padding:16px 20px; margin:26px auto 0; max-width:880px; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; align-items:start; }}
  .missing {{ color:var(--mut); font-style:italic; border:1px dashed #2a3157; border-radius:8px; padding:16px; }}
  .navbar {{ position:fixed; bottom:0; left:0; right:0; background:#0c0f1faa; backdrop-filter:blur(8px);
             border-top:1px solid #2a3157; padding:8px; text-align:center; }}
  .navbar button {{ background:#1a1f38; color:var(--ink); border:1px solid #2a3157; border-radius:8px;
                    margin:2px; padding:5px 10px; cursor:pointer; font-size:13px; }}
  .navbar button:hover {{ border-color:var(--acc); }}
  @media (max-width:760px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style></head><body>
<div class="deck">{slides_html}</div>
<div class="navbar">{nav}</div>
<script>
  const slides=[...document.querySelectorAll('.slide')];
  let cur=0;
  function go(i){{cur=Math.max(0,Math.min(slides.length-1,i));
    slides[cur].scrollIntoView({{behavior:'smooth',block:'start'}});}}
  document.addEventListener('keydown',e=>{{
    if(e.key==='ArrowRight'||e.key==='PageDown'){{go(cur+1);}}
    if(e.key==='ArrowLeft'||e.key==='PageUp'){{go(cur-1);}}
  }});
</script></body></html>"""


def main():
    OUT.write_text(render())
    print("wrote", OUT, f"({OUT.stat().st_size//1024} KB, {len(SLIDES)} slides)")


if __name__ == "__main__":
    main()

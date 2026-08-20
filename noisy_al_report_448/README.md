# Noisy-label active-learning report (448 unique runs)

This folder is a self-contained LaTeX report generated from
`data/original_features_456_runs.csv`. The analysis removes eight duplicated
ordinary MaxHerding rows, leaving 448 unique completed configurations.

## Regenerate the data and figures

```bash
python3 generate_assets.py
```

The script asserts the expected row counts and writes summary CSVs, LaTeX table
fragments, vector PDF figures, and PNG previews.

## Build the report

```bash
tectonic main.tex --outdir output/pdf
mv output/pdf/main.pdf output/pdf/noisy_label_active_learning_report.pdf
```

The final deliverable is `output/pdf/noisy_label_active_learning_report.pdf`.
The report was verified with Tectonic 0.16.9. A regular TeX Live installation
can also build it with `latexmk -pdf main.tex`.

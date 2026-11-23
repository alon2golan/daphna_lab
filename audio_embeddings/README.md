# Audio embeddings

This package creates dataset manifests, extracts audio embeddings, and runs audio active-learning experiments with SSAST and BEATs. Acquisition strategies live in `acquisition/` and consume global IDs plus the feature geometry supplied by the caller. Implementations are organized under `data/`, `models/`, `training/`, `evaluation/`, and `extraction/`; the package CLI and standalone AL command remain the user-facing entry points. `scan/` remains separate. Install audio dependencies in their own environment. Dataset audio, checkpoints, caches, and outputs should stay outside Git.

## Environment and checkpoints

Requires Python 3.10 or newer. Create the environment from the repository root:

```powershell
py -m venv .venv-audio
.venv-audio\Scripts\Activate.ps1
python -m pip install -r audio_embeddings\requirements.txt
```

The main base-models supported are BEATs and SSAST.
BEATs and SSAST source code is vendored under `audio_embeddings/models_src/` and selected automatically by their adapters. The adapters accept `--model-source` only when you intentionally want to point at another source checkout. BEATs uses its official checkpoint configuration and strict state-dict loading. SSAST uses the source-required `timm==0.4.5`, official ASTModel loader, and validates encoder key coverage. The ESC-50 configs point to the repository's included `BEATs_iter3_plus_AS2M.pt` and `SSAST-Base-Frame-400.pth` checkpoints; the SSAST frame checkpoint is adapted by the official ASTModel to the configured input shape. Each config still requires a provenance JSON describing the checkpoint and confirming held-out evaluation audio exclusion before a run can start. If a checkpoint's pretraining overlap cannot be established, do not mark that field as true. For CLAP use a compatible official LAION CLAP non-fusion checkpoint; for PANNs use a compatible official Cnn14 checkpoint. No checkpoint is downloaded implicitly.

Install the SSAST-pinned timm version when using SSAST:
```powershell
python -m pip install timm==0.4.5
```
SSAST and BEATs experiments require `model.checkpoint_metadata` pointing to a JSON object with `source`, `pretraining_data`, `license`, `evaluation_audio_excluded: true`, and `supervised_finetuning: false`. These fields are copied into immutable `run.json`; the last two are checked before extraction starts.

Compact audio batches use `.tar.zst` archives and require `zstandard` (included in `requirements.txt`). Keep dataset archives, feature caches, and run outputs outside ordinary Git history. Use approved shared storage or Git LFS for remote transfer; compression savings depend on the source audio codec.

## Manifest format

Manifests are UTF-8 JSON Lines, one sample per row. Required fields are `sample_id`, `audio_path` (nullable for unavailable source audio), `dataset`, `dataset_version`, `labels` (native scalar/object/list), and `split`. Dataset-specific metadata such as `fold`, `speaker_id`, `source_video_id`, and clip timestamps is retained on each row. IDs must be unique. Paths may be absolute or relative to the current working directory.

Build manifests with `python -m audio_embeddings manifest --dataset NAME --root DATA_ROOT --output PATH`. 
Supported dataset names include `fsd50k`, `urbansound8k`, `esc50`, `as20k`, DCASE 2016 Task 1 and Task 4 aliases, Speech Commands v1/v2 (`ks1`/`ks2`), and emotion-recognition aliases `er`/`iemocap`. The CLI accepts the exact aliases listed by `python -m audio_embeddings manifest --help`.

Examples:

```powershell
python -m audio_embeddings manifest --dataset fsd50k --root D:\datasets\FSD50K --output D:\manifests\fsd50k.jsonl
python -m audio_embeddings manifest --dataset urbansound8k --root D:\datasets\UrbanSound8K --output D:\manifests\urbansound8k.jsonl
python -m audio_embeddings manifest --dataset esc50 --root D:\datasets\ESC-50 --output D:\manifests\esc50.jsonl
python -m audio_embeddings manifest --dataset as20k --root D:\datasets\AudioSetAudio --metadata D:\datasets\AudioSetMetadata --output D:\manifests\as20k.jsonl
python -m audio_embeddings manifest --dataset ks2 --root D:\datasets\speech_commands_v2 --output D:\manifests\ks2.jsonl
```

FSD50K expects `ground_truth/dev.csv`, `ground_truth/eval.csv`, and extracted dev/eval audio directories; the builder also recognizes the dataset's `FSD50K.ground_truth`, `FSD50K.dev_audio`, and `FSD50K.eval_audio` directory names. It reads the official `split` column in `dev.csv` as train/validation and labels `eval.csv` as evaluation, preserving both label names and hierarchical `mids`. When `FSD50K.metadata/{dev,eval}_clips_info_FSD50K.json` exists, per-clip license values are retained. UrbanSound8K expects `metadata/UrbanSound8K.csv` and `audio/foldN/`. ESC-50 expects `meta/esc50.csv` and `audio/`. AudioSet metadata requires `balanced_train_segments.csv`, `eval_segments.csv`, and `class_labels_indices.csv`; audio is expected as `<split-folder>/<video>_<start>_<end>.wav`. The `as20k` command uses balanced training segments as the AL pool and evaluation segments as held-out test.

DCASE aliases `dcase`, `dcase2016`, and Task 1 spellings (`dcase2016_1`, `dcase20161`, including hyphenated equivalents) mean **DCASE 2016 Task 1: TUT Acoustic Scenes**, a multiclass task. Its normalized metadata CSV requires `recording_id,audio_path,scene,fold`; optional `official_split` or `split` values are retained. Use `dcase2016_4`, `dcase20164`, or explicit hyphenated equivalents only for **Task 4**, the multilabel sound event tagging task. Task 4 metadata uses `recording_id,audio_path,scene,fold,event_class,onset,offset`, one event per row. Provide `official_split` (or `split`) as `development`/`evaluation` when available. Supply `duration_seconds` whenever audio is unavailable so empty windows remain represented. Task 4 manifests use non-overlapping 10-second windows, multi-label targets, and relative event boundaries. Audio paths may be absolute or relative to `--root`.

Speech Commands expects the extracted version root with `validation_list.txt` and `testing_list.txt`. Those official lists define validation and test; remaining WAV files are train. The v1 manifest retains native labels and speaker IDs and labels the task as keyword/unknown/silence; silence examples are task-generated and are not fabricated in the source manifest. v2 retains its 35 native keyword classes. Use the dataset release documentation to select the version.

## Extract

```powershell
python -m audio_embeddings extract --manifest D:\manifests\esc50.jsonl --model beats --checkpoint D:\checkpoints\BEATs_iter3_plus_AS2M.pt --batch-size 16 --device cuda:0 --output D:\audio-embeddings
python -m audio_embeddings extract --manifest D:\manifests\esc50.jsonl --model clap --checkpoint D:\checkpoints\630k-audioset-best.pt --device cuda:0 --output D:\audio-embeddings
python -m audio_embeddings extract --manifest D:\manifests\esc50.jsonl --model panns --checkpoint D:\checkpoints\Cnn14_mAP=0.431.pth --device cuda:0 --output D:\audio-embeddings
```

BEATs and SSAST source code is discovered in the vendored `models_src/` tree by default; pass `--model-source` only when using another source checkout. SSAST extraction also requires `--checkpoint-metadata` with the provenance JSON described above.

### Compact audio batches and SLURM extraction

Package audio into independent compressed archives before submitting extraction. The default archive has up to 500 samples; `extraction.batch_size` is a separate model-memory setting and defaults to 16. Each extraction task expands one archive into a temporary directory, processes it in model batches, saves its output, validates it, and removes the expanded temporary files. The compressed archives remain for retries.

```bash
python -m audio_embeddings manifest \
  --dataset esc50 --root /shared/datasets/ESC-50 \
  --output /shared/audio/esc50/esc50.jsonl

python -m audio_embeddings pack-audio \
  --manifest /shared/audio/esc50/esc50.jsonl \
  --output /shared/audio/esc50/compact --batch-size 500

python -m audio_embeddings extract-array \
  --manifest /shared/audio/esc50/esc50.jsonl \
  --audio-index /shared/audio/esc50/compact/audio_index.json \
  --model ssast \
  --checkpoint /shared/audio/checkpoints/SSAST-Base-Frame-400.pth \
  --checkpoint-metadata /shared/audio/checkpoints/ssast_pretrained.provenance.json \
  --output /shared/audio/features/esc50-ssast \
  --job-dir /shared/audio/slurm/esc50-ssast-extract \
  --batch-size 16 --device cuda:0 --gres gpu:1 \
  --temporary-root /scratch/audio-extract \
  --activate /shared/envs/audio/bin/activate \
  --working-directory /shared/projects/daphna_lab

EXTRACT_JOB=$(sbatch --parsable /shared/audio/slurm/esc50-ssast-extract/audio-extract-array.sh)

cat > /shared/audio/slurm/finalize-esc50-ssast.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd /shared/projects/daphna_lab
source /shared/envs/audio/bin/activate
python -m audio_embeddings finalize-extract \
  --manifest /shared/audio/esc50/esc50.jsonl \
  --audio-index /shared/audio/esc50/compact/audio_index.json \
  --output /shared/audio/features/esc50-ssast
EOF
FINALIZE_JOB=$(sbatch --parsable --dependency="afterok:${EXTRACT_JOB}" \
  /shared/audio/slurm/finalize-esc50-ssast.sh)
```

All paths in an extraction array must be visible on every participating node. Each task writes a separate archive result. Use a fresh extraction output directory when changing the checkpoint, source, preprocessing, device, or model batch size. If some tasks fail, resubmit those array indices; do not finalize until all indexed archives have valid outputs. After successful finalization, the feature cache is at `/shared/audio/features/esc50-ssast/feature_cache`.

Frozen-representation and linear-probe AL jobs consume that cache and do not expand audio. Fine-tuning still needs audio paths: the all-method example below expands its archives to the configured temporary root for the `cls_al` job, retains them during training, and removes them after successful completion. Keep compressed archives until the workflow has completed and you no longer need retry input.

`run-al` reads `device` from YAML (`cpu`, `cuda`, or `cuda:<index>`). Override it for one invocation with `--device`, for example:

```powershell
python -m audio_embeddings run-al --config audio_embeddings\configs\esc50\ssast_core_set.yaml --device cuda:0
```

Ready ESC-50 configs are in `audio_embeddings/configs/esc50/`: one file for each supported method (`random`, `uncertainty`, `min_margin`, `max_entropy`, `bald`, `core_set`, `probcover`, `max_herding`, `typiclust`, `cls_al`) crossed with SSAST and BEATs. Each method file inherits a model-specific base file. Edit the manifest, checkpoint/provenance, and output paths for your machine. `min_margin` aliases `margin`, and `max_entropy` aliases `entropy`. CLS-AL selects `fine_tune`; the other acquisition configs use frozen representations. Percent budgets are cumulative global pool query counts.

### Acquisition geometry

Selectors do not normalize features or read labels. The ESC-50 base configs explicitly set `acquisition.feature_transform: l2`; use `identity` to retain raw encoder geometry. CoreSet uses squared Euclidean farthest-first distance, ProbCover uses Euclidean radius `delta`, and MaxHerding uses an RBF kernel with bandwidth `sigma` (`delta` remains accepted as a compatibility alias). TypiClust clusters only current candidates into as many clusters as the query budget and selects each cluster's most locally dense item; local density is the mean distance to up to 20 nearest other members, with singleton density treated as infinite. If duplicate features cause empty k-means clusters, remaining budget slots are filled by ascending global candidate ID.

Each run gets a configuration-derived subdirectory. `embeddings.npz` contains aligned `sample_ids` and a float32 `embeddings` matrix; `samples.jsonl` contains labels/splits in the exact same row order; `failures.json` records unavailable or unreadable samples. `run.json` records checkpoint SHA-256, manifest SHA-256, model and adapter identity, device, channel handling, sample rate, pooling/window policy, embedding dimension, and runtime versions. Re-running the same manifest/checkpoint/settings resumes missing samples. A different manifest, checkpoint, model, batch size, or device creates a separate run.

The AL runner writes per-seed/fold `scores.csv`, setting-aware aggregate `score_summary.csv` and `score_summary.md`, and task-specific metric plots with uncertainty bars and low-budget panels. State and result files are replaced atomically. Fine-tuning saves resumable model, optimizer, RNG, and progress state every five epochs and at each completed AL budget. Resume state is checked against the manifest, pretrained checkpoint, task, configuration, fold, seed, and method. By default the runner also fits a full-pool frozen-classifier reference (`report_full_label_reference: true`); these rows are marked `full_label_supervised_reference` and kept separate from AL methods. Disable it when a full-pool reference is not needed.

ProbCover accepts a positive configured `acquisition.delta`, or `acquisition.delta: auto`. Automatic selection reuses the deep_al representation-space heuristic, using seeded k-means pseudo-labels, a purity curve, and its inflection point; the selected radius and runtime are included in sampler scores and run metadata.

For SLURM arrays, add a `sweep` mapping to the experiment YAML with either a `grid` of dotted config keys to cross or a `jobs` list of explicit dotted-key overrides. Use `jobs` when methods need different valid adaptation modes. Run `python -m audio_embeddings sweep --config <config.yaml> --job-dir <external-dir> --dependency "afterok:<finalize-job-id>" --submit` to submit an array that starts only after feature finalization succeeds. `sweep.slurm` accepts SBATCH resources plus optional `working_directory` and environment `activate` script. Every generated job config records its sweep index and parameter values. Sweep arrays run independent AL configurations; they do not shard extraction.

### End-to-end ESC-50 / SSAST five-method linear-probe sweep

The example config [ssast_linearprobe_five_methods_sweep.yaml](configs/esc50/ssast_linearprobe_five_methods_sweep.yaml) creates five independent linear-probe jobs for MaxHerding, ProbCover, Margin, Uncertainty, and TypiClust. All five jobs read the same finalized SSAST feature cache, manifest, seeds, folds, and budgets. Since they use linear probes, they do not expand the audio archives. Update the shared paths inherited from the all-method config and provide accurate checkpoint provenance before submitting.

After the manifest, compact archives, and SSAST feature cache are prepared as shown above, submit the AL array dependent on extraction finalization:

```bash
python -m audio_embeddings sweep \
  --config audio_embeddings/configs/esc50/ssast_linearprobe_five_methods_sweep.yaml \
  --job-dir /shared/audio/slurm/esc50-ssast-five-methods \
  --dependency "afterok:${FINALIZE_JOB}" --submit
```

For the complete method set, use [ssast_all_methods_sweep.yaml](configs/esc50/ssast_all_methods_sweep.yaml) with the same command and a distinct `--job-dir`. It submits ten configured selectors: `random`, `uncertainty`, `min_margin`, `max_entropy`, `bald`, `core_set`, `probcover`, `max_herding`, `typiclust`, and `cls_al`. `least_confidence` aliases `uncertainty`; `margin` and `entropy` alias `min_margin` and `max_entropy`. The all-method config uses linear probes where supported, a dropout-enabled frozen-representation head for BALD, and fine-tuning for CLS-AL. The CLS-AL job expands compressed audio temporarily and removes expanded files after successful completion. Each sweep job writes to a distinct configuration-derived run directory. Use `seeds: [42, 43, 44, 45, 46]` and all five folds for a comparable ESC-50 evaluation; reduce them only for a smoke run.

Audio loading and resampling use torchaudio only. Audio is converted to mono by channel averaging and resampled with `torchaudio.functional.resample`. BEATs uses 16 kHz; CLAP uses its official 48 kHz waveform embedding API; PANNs uses the official 32 kHz inference embedding. These adapters use non-overlapping ten-second windows and zero-pad the last short window. SSAST uses 16 kHz and the official 128-bin Kaldi log-mel filterbank, SSAST normalization constants (-4.2677393 mean, 4.5689974 standard deviation with the official x2 scale), and exactly 1024 frames/10.24 seconds. SSAST averages the CLS and distillation tokens before its downstream task head. DCASE rows are cropped to their start/end boundaries before feature computation.

## Evaluation and licensing

Extraction does not train or tune a model. Downstream evaluation must use each manifest's `split`/`fold` protocol and must keep evaluation data held out. Task 1 uses multiclass scene metrics. Task 4 mAP and segment ER/F-score are measured from 10-second window tags; thresholded tags are expanded over each window for `sed_eval` at one-second resolution, so these predictions do not support event-timing F-score. FSD50K per-clip license metadata is retained when present; check the source license for every dataset and model checkpoint. AudioSet uses YouTube source segments and may have unavailable audio; failures are recorded without changing its split assignment.

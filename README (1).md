# EEG Feature Benchmark for Continuous Wrist Trajectory Decoding

Benchmark of EEG feature-extraction frontends and of GCN graph structures for
continuous wrist trajectory decoding on WAY-EEG-GAL (12 subjects, grasp-and-lift).
The model backbone is kept fixed, so any difference in score can be attributed to
the feature or to the adjacency alone.

## 1. Pipeline

```
feature -> GCN -> UniMamba3-A3 -> head -> R2 / RMSE
```

- **feature**: precomputed per trial, shape `[1, t, C, F]` (patches x channels x feature dim)
- **GCN**: `PrecomputedFeaturePatchGCN` (`handband_kalman.py`), assigned to `model.patch`.
  Graph structure comes from a fixed `[32, 32]` adjacency; only `feat_dim` changes
  between features.
- **backbone**: `build_unimamba3_bottleneck` (A3 / `ScanBottleneck`), identical for
  every condition: obs bottleneck -> lift -> Mamba blocks -> norm
- **head**: applied over the full sequence, one prediction per patch

Protocol is **within-subject**: train on sessions 1-6, validate on 7, test on 8-9.
Target normalisation statistics (`ym`, `ys`) and the adjacency are both computed on
the training split only. Validation is consumed by early stopping and model
selection; all reported figures come from the held-out test split. R2 is computed
per trial and then averaged, since raw error is not comparable across trials with
different position ranges.

## 2. Features

Ten conditions are benchmarked per subject.

| Feature | dim | What it captures |
|---|---|---|
| raw eeg | 125 | Full signal, no extraction. Floor baseline. |
| filtered | 5 | Band-pass value at patch centre (local phase). |
| bandpower | 5 | Log band power. Mu/beta ERD, the established motor correlate. |
| linelength | 5 | Mean absolute first difference. Cheap activation measure. |
| zerocross | 5 | Zero-crossing rate. Amplitude-free frequency estimate. |
| handfeat | 20 | The four above combined. Ceiling for amplitude features. |
| phase-locking value | 5 | Inter-channel phase synchronisation (connectivity). |
| analytic signal | 10 | Re/Im per band: amplitude and local phase together. |
| handfeat + plv | 25 | Does connectivity add anything on top of all amplitude features? |
| bandpower + plv | 10 | Same question against the single established motor correlate. |

All amplitude features are sliced from the same `BandMultiFeatureFrontend` output,
so preprocessing is identical across conditions.

PLV is computed per patch and per band: zero-phase band-pass filtering, Hilbert
transform, unit phase vectors, then the modulus of the time-averaged phase
difference over the 125 samples of each patch, for all channel pairs at once.
`compute_plv` returns the full `[t, n_bands, C, C]` matrix; the benchmark feature
is obtained by averaging each channel's synchronisation with the others.

## 3. Graph structures

The adjacency is a dataset-wide fixed `[32, 32]` matrix, not a per-trial quantity:
it is the graph over which the GCN aggregates, and it must stay constant so that
every training run shares the same structure. All modes are built the same way,
from the training trials of all subjects, and saved under the key `grand_corr` so
that they load through the same path. For each subject, trials are filtered
individually, their second-order statistics accumulated, and one matrix produced;
subject matrices are then averaged so every subject weighs equally.

| Mode | Measure | Band |
|---|---|---|
| `cov` | amplitude covariation | broadband (already low-passed at 40 Hz) |
| `cov_beta` | amplitude covariation | 13-30 Hz |
| `plv_beta` | phase synchronisation | 13-30 Hz |
| `plv_delta` | phase synchronisation | 1-4 Hz |
| `plv_alpha` | phase synchronisation | 8-13 Hz |

These are not five competing designs but a set of controls. An earlier run showed
that a beta-band PLV adjacency outperforms a broadband covariance one, but those
two differ in **two** respects at once - the measure and the frequency band - so
neither can be credited. The modes above separate them:

- `cov` vs `cov_beta`: measure held fixed, band varies
- `cov_beta` vs `plv_beta`: band held fixed, measure varies
- `plv_delta` / `plv_alpha` vs `plv_beta`: measure held fixed, band varies

`plv_delta` is the negative control: delta carries no motor rhythm, so if a delta
PLV graph performed as well as a beta one, the advantage could not be attributed
to motor physiology and would more likely come from numerical properties of the
matrix (the PLV is bounded in [0, 1] and strictly positive, the correlation is
signed and more unevenly spread). `plv_alpha` is exploratory rather than a
control, since mu is also a sensorimotor rhythm.

`cov` also serves as a bridge to the precomputed covariance matrix used in
earlier runs: it is computed from the same trials as everything else, so if it
reproduces those scores, the difference in data pipeline is negligible.

## 4. How to run

Place `run_benchmark.py` in the `scripts/` folder on **inlsrv4**, next to
`train_stream_n7_handfeat.py`. The hardcoded paths match the environment
`handband_kalman.py` was tested on, and the import of `train_stream_n7_handfeat`
must come first because it sets up `sys.path` for the `models` imports.

```bash
python run_benchmark.py                          # broadband covariance (default)
python run_benchmark.py --adjacency cov_beta     # beta-band covariance
python run_benchmark.py --adjacency plv_beta     # beta-band PLV
python run_benchmark.py --adjacency plv_delta    # delta-band PLV (negative control)
python run_benchmark.py --adjacency plv_alpha    # alpha-band PLV
```

Each run first builds its adjacency from the training trials (a few minutes),
saves it to `OUT_DIR/adj_<mode>.npz`, then runs the benchmark.

Output: `benchmark_<mode>.csv` in `OUT_DIR`, one row per (subject, feature) with
`r2_mean`, `r2_median` and `rmse`. The CSV is rewritten after each subject, so an
interrupted run still leaves usable results. Since runs differ only by the
adjacency, their CSVs form a paired design: for a given subject and feature, the
difference between two files isolates the effect of the graph, and a paired test
across subjects can be run on those differences.

**Paths to check before running** (both hardcoded at the top of the script):

- `ADJ_NPZ` -> `/home/hdeng/PD_gait_decoding/progress/N12_streaming_gcn/channel_covariance_32.npz`
- `OUT_DIR` -> `/home/hdeng/PD_gait_decoding/Paper/06_Mamba/B_BrainSignal/normalize_ablation/results`

Requirements: torch, numpy, scipy, pandas. A GPU is expected; each run is roughly
one training per (subject, feature), so about 120 trainings.

## 5. Known limitations

- All results come from a single seed. Whether the differences survive across
  seeds still has to be checked.
- PLV estimated over 125 samples has a positive noise floor; independent channels
  do not give exactly zero. Absolute values should be read with that in mind.
- The correlation adjacencies are signed, the PLV ones are not. A negative weight
  makes the GCN subtract a neighbour's features rather than add them, which is a
  different propagation regime and a confound none of the current controls
  isolates.
- A band-limited covariance is a poor estimator of electrode layout, which is
  what a covariance adjacency is normally for. It is included as a control, not
  as a candidate design.
- In `handfeat + plv` and `bandpower + plv`, the two blocks are concatenated on
  their raw scales (log-power values vs PLV in [0, 1]) without prior normalisation.
- AR coefficients, wavelet features, CSP and Riemannian representations were left
  out, either as redundant with the spectral features already included or as
  poorly suited to the patch-based pipeline.

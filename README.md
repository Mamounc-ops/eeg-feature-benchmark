# EEG Feature Benchmark for Continuous Wrist Trajectory Decoding

Benchmark of EEG feature-extraction frontends for continuous wrist trajectory
decoding on WAY-EEG-GAL (12 subjects, grasp-and-lift). The model backbone is
kept fixed so that any difference in score can be attributed to the feature
alone.

## 1. Pipeline

```
feature → GCN → UniMamba3-A3 → head → R² / RMSE
```

- **feature**: precomputed per trial, shape `[1, t, C, F]` (patches x channels x feature dim)
- **GCN**: `PrecomputedFeaturePatchGCN` (`handband_kalman.py`), assigned to `model.patch`.
  Graph structure comes from a fixed `[32, 32]` adjacency; only `feat_dim` changes
  between features.
- **backbone**: `build_unimamba3_bottleneck` (A3 / `ScanBottleneck`), identical for
  every condition: obs bottleneck → lift → Mamba blocks → norm
- **head**: applied over the full sequence, one prediction per patch

Protocol is **within-subject**: train on sessions 1-6, validate on 7, test on 8-9.
Target normalisation statistics (`ym`, `ys`) are computed on the training split only.
Validation is consumed by early stopping and model selection; all reported figures
come from the held-out test split. R² is computed per trial and then averaged, since
raw error is not comparable across trials with different position ranges.

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

## 3. Bonus: covariance vs PLV adjacency

The GCN adjacency is currently channel correlation, which only captures linear
relationships. The same benchmark can be run with a PLV adjacency instead, keeping
everything else fixed. This gives a paired design: for each (subject, feature) pair
there are two scores differing only by the graph structure, so the per-pair
difference isolates the effect of the adjacency.

In PLV mode, a single global `[32, 32]` matrix is computed by averaging beta-band
PLV over all training trials of all subjects, then saved to `OUT_DIR/adj_plv.npz`
under the key `grand_corr` and loaded exactly like the covariance matrix. Like the
covariance, it is a dataset-wide fixed graph structure, not a per-trial quantity.
It is built from training sessions only.

## 4. How to run

Place `run_benchmark.py` in the `scripts/` folder on **inlsrv4**, next to
`train_stream_n7_handfeat.py`. The hardcoded paths match the environment
`handband_kalman.py` was tested on, and the import of `train_stream_n7_handfeat`
must come first because it sets up `sys.path` for the `models` imports.

```bash
# covariance adjacency (default)
python run_benchmark.py

# PLV adjacency, computed on the fly before the benchmark starts
python run_benchmark.py --adjacency plv
```

Output: `benchmark_cov.csv` or `benchmark_plv.csv` in `OUT_DIR`, one row per
(subject, feature) with `r2_mean`, `r2_median` and `rmse`. The CSV is rewritten
after each subject, so an interrupted run still leaves usable results.

**Paths to check before running** (both hardcoded at the top of the script):

- `ADJ_NPZ` → `/home/hdeng/PD_gait_decoding/progress/N12_streaming_gcn/channel_covariance_32.npz`
- `OUT_DIR` → `/home/hdeng/PD_gait_decoding/Paper/06_Mamba/B_BrainSignal/normalize_ablation/results`

Requirements: torch, numpy, scipy, pandas. A GPU is expected; the run is roughly
one training per (subject, feature), so about 120 trainings for the full grid.
PLV mode adds a few minutes upfront to build the adjacency.

## 5. Known limitations

- PLV estimated over 125 samples has a positive noise floor; independent channels
  do not give exactly zero. Absolute values should be read with that in mind.
- In `handfeat + plv` and `bandpower + plv`, the two blocks are concatenated on
  their raw scales (log-power values vs PLV in [0, 1]) without prior normalisation.
- Covariance adjacency only captures linear relationships and may create spurious
  edges through volume conduction. This is one motivation for the PLV comparison.
- AR coefficients, wavelet features, CSP and Riemannian representations were left
  out, either as redundant with the spectral features already included or as
  poorly suited to the patch-based pipeline.

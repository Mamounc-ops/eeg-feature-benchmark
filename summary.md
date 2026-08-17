# EEG feature benchmark for continuous wrist trajectory decoding — summary

## Key results

<!-- TODO: fill after the adjacency ablation completes.
     Five bullets max. Rule: every bullet naming PLV must state which axis it
     refers to, in the bullet itself, e.g.
       - PLV **as adjacency**: ...
       - PLV **as node feature**: ...
     Never write "PLV improves R²" without the qualifier. -->

## Introduction

Continuous wrist trajectory decoding from scalp EEG is limited less by the
temporal model than by what is fed into it. The backbone used here
(UniMamba3-A3) is already fixed by prior work in this project; the open question
is which representation of the 32-channel signal it should consume, and how the
channels should be coupled to each other before the temporal model sees them.

This report benchmarks two **orthogonal** design axes:

- **Axis 1 — the feature frontend.** How each (channel, patch) of raw EEG is
  turned into a node feature vector: band power, line length, zero-crossing
  rate, the analytic signal, phase-locking value, or combinations thereof.
- **Axis 2 — the GCN adjacency.** Which matrix $A \in \mathbb{R}^{32 \times 32}$
  defines the graph over which the spatial mixer aggregates: broadband
  covariance, band-limited covariance, or band-limited PLV.

These two axes are independent and must not be conflated. A quantity such as
PLV can appear on **either** axis, and it does not behave the same way on both:

| | Where it enters | What it does |
|---|---|---|
| **PLV as node feature** | Axis 1, frontend output | A per-channel descriptor concatenated into the token fed to the GCN |
| **PLV as adjacency** | Axis 2, GCN graph | Replaces the covariance matrix; sets *which channels talk to which* |

Every claim in this report is tagged with the axis it belongs to. A statement
about "PLV" with no axis tag is ambiguous and should be treated as a typo.

The benchmark was built in two stages, which explains the shape of the
experiment grid. Axis 1 was swept first, with adjacency held at the precomputed
broadband covariance matrix. Axis 2 was added afterwards, once the
frontend sweep showed that the choice of node feature mattered less than
expected. The grid is therefore not fully crossed; Section
[Results](#results) states explicitly which cells were run.

## Pipeline & evaluation protocol

### Fixed chain

Everything outside the two swept axes is held constant across all runs:

![Pipeline](figures/pipeline.svg)

```
EEG [B, T, 32]
  → frontend        (Axis 1)  → per (channel, patch) feature vector
  → GCN             (Axis 2)  → spatial mixing over the 32-node graph
  → linear lift     → embed_dim = 8
  → UniMamba3-A3    → 2 blocks, d_state = 8, expand = 2, headdim = 8,
                      rope_fraction = 0.5, causal (forward scan only)
  → Conv1d head     → wrist coordinate at the anchor sample
```

The GCN is a fixed-adjacency graph convolution,

$$x' = W_{\text{root}} \, x + W_{\text{rel}} \, (A x)$$

with the diagonal of $A$ zeroed. Note that $A$ is the **graph structure**, not a
node feature: it is never concatenated into $x$. Optional sparsification by
`--adj_threshold` or `--adj_topk` is available but is left off (dense $A$) for
all runs reported here.

### Signal conditioning

| Parameter | Value |
|---|---|
| Dataset | WAY-EEG-GAL, grasp-and-lift |
| Subjects | 12 |
| Channels | 32 |
| Sampling rate | 500 Hz |
| Low-pass | 40 Hz |
| Normalization | per-trial z-score |
| Patch size | 125 samples (250 ms) |
| Input window | 0.5 s (250 samples), stride 0.1 s |
| Head window | 2 patches |
| Trials per session | capped at 15 |

### Split and training

Sessions are split **within subject**: sessions 1–6 train, session 7
validation, sessions 8–9 test. Each subject gets its own model; there is no
cross-subject transfer. Training runs to 80 epochs max with early stopping
(patience 10) on validation, Adam at `lr = 1e-3`.

### Metric

Per test trial, over the anchor points of the streaming prediction,

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$R^2$ is averaged over trials, then over the 12 subjects. The reported number
is that grand mean; the median across trials and the mean RMSE (in the target's
physical units) are logged alongside it.

**Why $R^2$ and not MSE.** EEG is z-scored per trial, so the model's output
scale is restored via the training-set target statistics rather than being
intrinsic. Raw MSE is consequently not comparable across subjects — a subject
with a larger natural range of wrist motion produces a larger MSE at identical
decoding quality. $R^2$ normalizes by each trial's own variance, which makes
the across-subject mean meaningful. RMSE is retained only as an
interpretable per-subject side quantity, never as the ranking criterion.

**Seeds.** Results below are single-seed unless stated otherwise. See
[Discussion](#discussion) for what this does and does not license.

### Reproducing

```bash
python scripts/run_benchmark.py --subjects 1,2,3,4,5,6,7,8,9,10,11,12 \
    --frontend handfeat --adjacency plv_beta
```

`run_benchmark.py` imports `train_stream_n7_handfeat.py` first, which sets up
`sys.path` for the `models` package — that import order is load-bearing. Results
are appended to CSV after each subject, so a partial run is still usable.

## Benchmark axes

### Axis 1 — feature frontends

All frontends share the same band decomposition, five frozen `firwin` FIR
band-passes of length `patch_size`:

$$\delta \; 1{-}4, \quad \theta \; 4{-}8, \quad \alpha \; 8{-}13, \quad \beta \; 13{-}30, \quad \gamma_{\text{low}} \; 30{-}40 \;\; \text{Hz}$$

No frontend has learnable parameters; the only trainable layer between the
frontend and the backbone is the linear lift to `embed_dim = 8`. This keeps the
comparison about *what information is extracted*, not about how much capacity
sits in the extractor.

| Frontend | Dim | What it computes | Why it might carry wrist information |
|---|---|---|---|
| `raw` | 1 | Unfiltered sample at patch center | Baseline; the backbone must do all the work |
| `filtered` | 5 | Band-filtered value at patch center | Phase-preserving. Tests whether a frozen filter bank matches what a learned `PatchEmbed` finds |
| `bandpower` | 5 | $\log_{10}$ mean square per band | Sensorimotor $\mu$/$\beta$ desynchronization tracks movement onset and speed — the standard MI-EEG feature |
| `linelength` | 5 | $\log_{10}$ mean $\lvert \Delta x \rvert$ per band | Amplitude × frequency in one scalar; cheap on hardware and robust to baseline drift |
| `zerocross` | 5 | Sign-change rate per band | Dominant-frequency proxy, invariant to amplitude scaling |
| `analytic` | 10 | $(\mathrm{Re}, \mathrm{Im})$ of the analytic signal per band | Retains **both** amplitude and instantaneous phase, which the power-only features discard |
| `handfeat` | 20 | All four of the above per band | Tests whether the features are complementary or redundant |
| `plv_feat` | — | PLV of each channel against the montage, as a node feature | **Axis 1 use of PLV.** Gives each node a summary of how phase-consistent it is with the rest of the head |
| `handfeat_plv` | — | `handfeat` ⊕ `plv_feat` | Does phase consistency add anything on top of amplitude features? |
| `bandpower_plv` | — | `bandpower` ⊕ `plv_feat` | Same question against the strongest single-family baseline |

The motivation for including phase at all is that band power is a *rectified*
quantity: it discards the sign and timing of the oscillation. For a continuous
regression target such as wrist position — as opposed to a discrete
classification of movement vs. rest — the timing information may matter, and
`analytic` and `plv_feat` are the two ways of putting it back.

### Axis 2 — GCN adjacency

The original configuration used a precomputed broadband covariance matrix
(`channel_covariance_32.npz`). Substituting a $\beta$-band PLV matrix changed
performance substantially. That single comparison, however, confounds two
things at once:

1. the **measure** — covariance (amplitude coupling) vs. PLV (phase coupling);
2. the **band** — broadband vs. $\beta$ only.

The adjacency sweep is designed to separate them:

| `--adjacency` | Matrix | Role in the design |
|---|---|---|
| `cov` | Broadband covariance | Bridge to the precomputed matrix; the original baseline |
| `cov_beta` | $\beta$-band covariance | **Isolates the measure.** Same band as `plv_beta`, different coupling statistic. If `cov_beta` closes most of the gap to `plv_beta`, the effect was band selection, not phase |
| `plv_beta` | $\beta$-band PLV | The configuration that improved results |
| `plv_delta` | $\delta$-band PLV | **Negative control.** $\delta$ carries no sensorimotor rhythm. If it also works, the mechanism is numerical (conditioning of the normalized graph operator), not physiological |
| `plv_alpha` | $\alpha$-band PLV | Exploratory. $\mu$ is sensorimotor too, so this should partially work if the story is band-specific motor synchrony |

Note that `plv_delta` is the load-bearing cell. Every "PLV captures motor
synchrony" explanation predicts it fails; every "PLV is a better-conditioned
operator regardless of content" explanation predicts it succeeds. It is the
cheapest experiment that can falsify the physiological reading.

A caveat on framing: PLV is sometimes justified as *avoiding volume
conduction*. That is imprecise. PLV is computed from instantaneous phases and
remains sensitive to zero-lag contributions; the estimators actually designed to
suppress them are the imaginary part of coherency, PLI, and wPLI. The two
hypotheses tested here are stated instead in mechanistic, falsifiable terms:

- **H1 — amplitude invariance.** Covariance entries scale with channel variance,
  so the graph is dominated by high-power channels rather than by coupled ones.
  PLV is bounded in $[0, 1]$ and encodes only phase relationship.
- **H2 — band specificity.** `plv_beta` is band-limited while `cov` is
  broadband, so the comparison is not information-matched. `cov_beta`
  discriminates H1 from H2 directly.

## Results

<!-- TODO
     4.1  Feature × adjacency grid. State which cells were actually run.
     4.2  The interaction — this is the headline, not the two marginals.
     4.3  Seed stability.
     Figures: figures/grid_r2.pdf, figures/adjacency_heatmaps.pdf,
              figures/adjacency_eigenspectra.pdf
     All regenerated by scripts/make_figures.py from the benchmark CSVs. -->

## Discussion

<!-- TODO
     - Which of H1 / H2 the cov_beta cell supports.
     - What plv_delta implies (physiological vs. numerical).
     - Why PLV helps as adjacency but not as node feature — the asymmetry is
       the interesting result and needs an explicit account.
     - Limitations: single seed; grid not fully crossed; 12 subjects;
       within-subject only, no transfer claim. -->

## Conclusion

<!-- TODO: 3-4 sentences + concrete next steps (per-band covariance sweep,
     multi-seed rerun of the top cells). -->

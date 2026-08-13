
import sys, os, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_stream_n7_handfeat as TT         
from models.tinymyo_unimamba3 import build_unimamba3_bottleneck   
from models.handband_kalman import PrecomputedFeaturePatchGCN, BandMultiFeatureFrontend, BandAnalyticFrontend
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert


OUT_DIR = Path("/home/hdeng/PD_gait_decoding/Paper/06_Mamba/B_BrainSignal/normalize_ablation/results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SESSIONS = [1, 2, 3, 4, 5, 6]
VAL_SESSIONS = [7]
TEST_SESSIONS = [8, 9]
SUBJECTS = list(range(1, 13))
INPUT_WIN_S = 0.5
STRIDE_S = 0.1
FS = 500
INPUT_LEN = int(INPUT_WIN_S * FS)
LOWPASS_HZ = 40.0
NORMALIZE_MODE = "per_trial"
TARGET = "tail-last"
N7_PATCH = 125   
WIN = 2         
MAX_TRIALS_PER_SESSION = 15  
MAX_EPOCHS = 80
PATIENCE = 10
LR = 1e-3
BATCH = 64       
ADJ_NPZ = "/home/hdeng/PD_gait_decoding/progress/N12_streaming_gcn/channel_covariance_32.npz"
BANDS = [(1,4), (4,8), (8,13), (13,30), (30,40)]

def Build_Model(frontend) : 

    """Build a UniMamba3-A3 model with a custom feature frontend.

    The A3 backbone (ScanBottleneck variant) is instantiated with the standard
    patch size and embedding dimension, then its default patch embedding is
    replaced by the supplied frontend. All downstream stages (obs, lift, Mamba
    blocks, norm, head) are left untouched, so the model backbone stays fixed
    across the whole benchmark.

    Parameters
    ----------
    frontend : nn.Module
        Feature frontend assigned to ``model.patch``. It must accept the tensor
        passed by the training loop and return ``(tokens, C, t)`` with tokens of
        shape ``[B, C*t, embed_dim]`` in channel-first order.

    Returns
    -------
    model : nn.Module
        UniMamba3-A3 model whose ``patch`` attribute is the supplied frontend.

    Examples
    --------
    >>> frontend = PrecomputedFeaturePatchGCN(in_ch=32, feat_dim=5,
    ...                                       embed_dim=8, adj_npz_path=ADJ_NPZ)
    >>> model = Build_Model(frontend)
    """

    model = build_unimamba3_bottleneck(patch_size=N7_PATCH, embed_dim=8)
    model.patch = frontend
    return model



def compute_plv(eeg, fs=500, patch_size=125, bands=BANDS, zero_diag=True):

    """Compute the Phase Locking Value between all channel pairs, per patch and band.

    The signal is band-pass filtered (zero-phase) in each frequency band, the
    analytic signal is obtained by Hilbert transform, and phases are reduced to
    unit complex vectors. Within each non-overlapping patch, the PLV between two
    channels is the modulus of the time-averaged phase difference, computed for
    all pairs at once through a matrix product.

    Parameters
    ----------
    eeg : ndarray of shape (T, C)
        Raw EEG for one trial: T time samples, C channels.
    fs : int, default=500
        Sampling frequency in Hz.
    patch_size : int, default=125
        Number of samples per patch. The PLV is averaged over these samples.
    bands : list of (float, float), default=BANDS
        Frequency bands as (low, high) cutoffs in Hz.
    zero_diag : bool, default=True
        If True, set the diagonal to zero. The self-PLV is mathematically 1,
        but zero is the usual convention when the matrix is used as a graph
        adjacency, where self-loops are handled separately.

    Returns
    -------
    plv : ndarray of shape (t, n_bands, C, C)
        PLV matrices, with ``t = T // patch_size``. Values lie in [0, 1]: close
        to 1 when two channels keep a stable phase difference, close to 0 when
        their phase difference drifts. Each matrix is symmetric.

    Notes
    -----
    With only ``patch_size`` samples per estimate, independent signals do not
    yield a PLV of exactly zero but a positive noise floor. This is inherent to
    short windows and should be kept in mind when interpreting absolute values.

    Examples
    --------
    >>> plv = compute_plv(trial['eeg'])          # (t, 5, 32, 32)
    >>> beta = plv[:, 3]                         # beta band only, (t, 32, 32)
    """

    T, C = eeg.shape
    t = T // patch_size
    n_bands = len(bands)
    plv = np.zeros((t, n_bands, C, C), dtype=np.float32)

    for b, (lo, hi) in enumerate(bands):
        bb, aa = butter(4, [lo, hi], btype='band', fs=fs)
        filt = filtfilt(bb, aa, eeg, axis=0)          
        analytic = hilbert(filt, axis=0)              
        z = analytic / (np.abs(analytic) + 1e-12)  

        for p in range(t):
            zp = z[p*patch_size:(p+1)*patch_size]     
            M = (zp.conj().T @ zp) / zp.shape[0]       
            plv[p, b] = np.abs(M)                      
        if zero_diag:
            plv[:, b][:, np.arange(C), np.arange(C)] = 0.0

    return plv

def compute_grand_plv(subjects, sessions, band_idx=3, fs=FS, patch_size=N7_PATCH):

    """Compute a single global PLV matrix to be used as a GCN adjacency.

    Trial-level PLV matrices are accumulated over every training trial of every
    subject, in one frequency band, and averaged into one matrix. This mirrors
    how the covariance adjacency is built: the graph structure is a fixed,
    dataset-wide property rather than something that varies per trial.

    Parameters
    ----------
    subjects : list of int
        Subject identifiers to accumulate over.
    sessions : list of int
        Sessions to load. Only training sessions should be passed, so that no
        information from validation or test data leaks into the graph structure.
    band_idx : int, default=3
        Index of the band in ``BANDS`` used for the adjacency. The default is
        the beta band, the most relevant one for sensorimotor activity.
    fs : int, default=FS
        Sampling frequency in Hz.
    patch_size : int, default=N7_PATCH
        Number of samples per patch, passed through to ``compute_plv``.

    Returns
    -------
    grand : ndarray of shape (C, C), dtype float32
        Symmetric PLV matrix averaged over all patches and trials, with a zero
        diagonal. Ready to be saved under the key ``grand_corr`` so that it can
        be loaded exactly like the precomputed covariance adjacency.

    Examples
    --------
    >>> grand = compute_grand_plv(SUBJECTS, TRAIN_SESSIONS)
    >>> np.savez(OUT_DIR / "adj_plv.npz", grand_corr=grand)
    """

    acc = None
    n = 0
    for subj in subjects:
        trials = TT.get_full_trials(subj, sessions)
        if not trials:
            continue
        for tr in trials:
            plv = compute_plv(tr['eeg'], fs=fs, patch_size=patch_size)  
            beta = plv[:, band_idx]                                     
            s = beta.sum(axis=0)                                        
            acc = s if acc is None else acc + s
            n += beta.shape[0]
    grand = (acc / max(n, 1)).astype(np.float32)                       
    np.fill_diagonal(grand, 0.0)
    return grand

def compute_grand_cov(subjects, sessions, fs=FS, mode="large"):

    """Compute a global channel-correlation matrix to be used as a GCN adjacency.

    For each subject, the training trials are filtered (optionally band-limited),
    centred, and their second-order statistics accumulated into a single
    correlation matrix. Subject matrices are then averaged, so every subject
    weighs equally regardless of how much data it contributes. This mirrors the
    way the reference covariance adjacency was built.

    Parameters
    ----------
    subjects : list of int
        Subject identifiers to accumulate over.
    sessions : list of int
        Sessions to load. Only training sessions should be passed, so that no
        information from validation or test data leaks into the graph structure.
    fs : int, default=FS
        Sampling frequency in Hz, used to design the band-pass filter.
    mode : {"large", "beta"}, default="large"
        Frequency content used. "large" keeps the signal as delivered by the
        loader, which is already low-passed at 40 Hz. "beta" band-passes it to
        13-30 Hz, matching the band used for the PLV adjacency, so that the two
        differ only by the nature of the measure and not by their frequency
        content.

    Returns
    -------
    grand : ndarray of shape (C, C), dtype float32
        Correlation matrix averaged over subjects. Ready to be saved under the
        key ``grand_corr``.

    Notes
    -----
    Trials are filtered individually before their statistics are accumulated;
    filtering concatenated trials would create artefacts at the joins.

    Examples
    --------
    >>> A = compute_grand_cov(SUBJECTS, TRAIN_SESSIONS, mode="beta")
    >>> np.savez(OUT_DIR / "adj_cov_beta.npz", grand_corr=A)
    """


    bb = aa = None
    if mode == "beta":
        bb, aa = butter(4, [13, 30], btype='band', fs=fs)   
    elif mode != "large":
        raise ValueError(f"unknown mode: {mode}")

    cov = []
    for subj in subjects:
        trials = TT.get_full_trials(subj, sessions)
        if not trials:
            continue
        acc = np.zeros((32, 32), dtype=np.float64)          
        n = 0
        for tr in trials:
            x = tr['eeg'] if bb is None else filtfilt(bb, aa, tr['eeg'], axis=0)
            x = x - x.mean(axis=0, keepdims=True)
            acc += x.T @ x
            n += x.shape[0]
        C = acc / max(n - 1, 1)                              
        d = np.sqrt(np.maximum(np.diag(C), 1e-12))
        cov.append(C / np.outer(d, d))                       
    return np.mean(cov, axis=0).astype(np.float32)
            



def n11_features_full(model, x):

    """Run the A3 backbone up to the normalised latent sequence.

    Reproduces the forward pass of the A3 bottleneck variant: the frontend
    produces channel-first tokens, which are reordered into one vector per time
    step, projected down to the physical observation bottleneck, lifted back to
    the Mamba width, scanned by the Mamba blocks and normalised. The regression
    head is deliberately left out so that callers can apply it over the full
    sequence and obtain one prediction per patch.

    Parameters
    ----------
    model : nn.Module
        A3 model whose ``patch`` attribute is the feature frontend.
    x : torch.Tensor
        Input passed to the frontend. With a precomputed-feature frontend this
        is a feature tensor of shape ``[B, t, C, F]``.

    Returns
    -------
    seq : torch.Tensor of shape (B, p, d_model)
        Normalised latent sequence, one vector per patch.
    p : int
        Number of patches, as reported by the frontend.

    Notes
    -----
    The reshape from ``[B, C*p, D]`` to ``[B, p, C*D]`` assumes channel-first
    token ordering. Feeding time-first tokens would silently mix channels and
    patches instead of raising an error.
    """

    B = x.size(0)
    tokens, C, p = model.patch(x)              
    D = model.patch.embed_dim
    seq = tokens.reshape(B, C, p, D).permute(0, 2, 1, 3).reshape(B, p, C * D)       
    seq = model.obs(seq)                       
    seq = model.lift(seq)                      
    for blk in model.blocks:
        seq = blk(seq)
    seq = model.norm(seq)
    return seq, p

def stream_predict_one_a3(model, feature):
    
    """Predict a normalised trajectory for one trial, one value per patch.

    The whole feature sequence is passed through the backbone in a single call
    and the regression head is applied over all time steps, yielding a smooth
    prediction per patch rather than one value per sliding window.

    Parameters
    ----------
    model : nn.Module
        Trained A3 model.
    feature : torch.Tensor of shape (1, t, C, F)
        Precomputed features for one trial, already patched.

    Returns
    -------
    anchors : list of int or None
        Sample index at which each prediction is anchored, namely the last
        sample of each patch. None if the trial is too short.
    preds : torch.Tensor of shape (p,) or None
        Normalised predictions, one per patch. None if the trial is too short.

    Examples
    --------
    >>> anchors, preds = stream_predict_one_a3(model, features[0])
    >>> true = trial['kin_x'][np.array(anchors)]
    """
    p = feature.shape[1]
    if p < 1:
        return None, None
    x = feature.to(DEVICE)
    seq, p = n11_features_full(model, x)
    h = seq.transpose(1, 2)
    h = model.head(h)
    preds = h.squeeze(0).squeeze(0)
    anchors = [(i * N7_PATCH - 1) for i in range(1, p + 1)]
    return anchors, preds

def eval_stream_a3(model, features, trials, ym, ys):

    """Evaluate a model on a set of trials and report R-squared and RMSE.

    For each trial, predictions are mapped back to the original units and
    compared with the true wrist positions at the anchor samples. A separate
    R-squared is computed per trial, then aggregated. Per-trial R-squared is
    preferred over a raw error because it stays comparable across trials and
    subjects despite differing position ranges.

    Parameters
    ----------
    model : nn.Module
        Trained A3 model.
    features : list of torch.Tensor
        Precomputed features, one ``[1, t, C, F]`` tensor per trial, aligned
        with ``trials``.
    trials : list of dict
        Trials holding the ground-truth ``kin_x`` array.
    ym, ys : float
        Mean and standard deviation used to denormalise predictions. They must
        be the statistics computed on the training set.

    Returns
    -------
    r2_mean : float
        Mean per-trial R-squared, or NaN if no trial could be evaluated.
    r2_median : float
        Median per-trial R-squared, more robust to outlying trials.
    rmse : float
        Mean per-trial RMSE, in the original units of the wrist position.

    Examples
    --------
    >>> r2_mean, r2_med, rmse = eval_stream_a3(model, test_feats, test_trials, ym, ys)
    """
    model.eval()
    r2s, rmses = [], []
    with torch.no_grad():
        for fe, tr in zip(features, trials):
            anchors, preds = stream_predict_one_a3(model, fe)
            if anchors is None:
                continue
            pred_real = preds.cpu().numpy() * ys + ym
            true = tr['kin_x'][np.array(anchors)].astype(np.float64)
            sse = ((true - pred_real) ** 2).sum()
            ssm = max(((true - true.mean()) ** 2).sum(), 1e-9)
            r2s.append(1 - sse / ssm)
            rmses.append(float(np.sqrt(((true - pred_real) ** 2).mean())))
    return (float(np.mean(r2s)) if r2s else float("nan"),
            float(np.median(r2s)) if r2s else float("nan"),
            float(np.mean(rmses)) if rmses else float("nan"))

def train_streaming_a3(model, features, train_trials, val_trials,
                       val_features, ym, ys, log_prefix=""):
    
    """Train an A3 model on one feature set and return its best version.

    Targets are cached once, normalised with the training statistics, then the
    model is trained trial by trial with an MSE loss. After every epoch the
    model is scored on the validation split; the weights achieving the best
    validation R-squared are kept and restored at the end, and training stops
    early once validation stops improving.

    Parameters
    ----------
    model : nn.Module
        Freshly built A3 model.
    features : list of torch.Tensor
        Training features, one ``[1, t, C, F]`` tensor per trial.
    train_trials, val_trials : list of dict
        Trials providing the ground-truth ``kin_x`` arrays.
    val_features : list of torch.Tensor
        Validation features, aligned with ``val_trials``. They must come from
        the validation split, not the training one.
    ym, ys : float
        Target mean and standard deviation computed on the training split.
    log_prefix : str, default=""
        Prefix prepended to progress lines.

    Returns
    -------
    model : nn.Module
        The model reloaded with the weights of its best validation epoch, not
        those of the last epoch.

    Notes
    -----
    Validation is consumed here to select the model, so it can no longer serve
    as an unbiased score. The figures reported by the benchmark are computed on
    the held-out test split instead.
    """

    torch.manual_seed(42)

    model = model.to(DEVICE).float()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    best_val = -float("inf"); best_state = None; best_ep = 0; patience = PATIENCE

    train_cache = []
    for fe, tr in zip(features, train_trials):
        p = fe.shape[1]                     
        if p < 1:
            continue
        fe_t = fe                             
        anchors = np.array([(i * N7_PATCH - 1) for i in range(1, p + 1)])
        tgt_t = torch.from_numpy(tr['kin_x'][anchors]).float()
        tgt_n = (tgt_t - ym) / ys
        train_cache.append((fe_t, tgt_n, p))

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(train_cache))
        ep_loss = 0.0; n_anch = 0; t0 = time.time()

        for idx in perm:
            fe_t, tgt_n, p = train_cache[idx]
            x = fe_t.to(DEVICE)               

            seq, p_ = n11_features_full(model, x)
            h = seq.transpose(1, 2)
            h = model.head(h)
            preds = h.squeeze(0).squeeze(0)

            tgt = tgt_n.to(DEVICE)
            loss = F.mse_loss(preds, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss) * len(tgt); n_anch += len(tgt)

        val_mean, val_med, val_rmse = eval_stream_a3(model, val_features, val_trials, ym, ys)
        ep_time = time.time() - t0
        improved = val_mean > best_val + 1e-6
        if improved:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep; patience = PATIENCE
        else:
            patience -= 1

        if (ep % 5 == 0) or improved or ep == 1 or patience <= 0:
            print(f"  {log_prefix} ep={ep:3d}  loss={ep_loss/max(n_anch,1):.4f}  "
                  f"val_mean={val_mean:+.4f} med={val_med:+.4f}  "
                  f"rmse={val_rmse:.2f}  ({ep_time:.1f}s) "
                  f"{'★' if improved else ''}", flush=True)
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def main(adjacency = "cov"):

    """Run the full feature benchmark across subjects and write results to CSV.

    For every subject, trials are loaded, target statistics are computed on the
    training split, and every feature is extracted for the three splits. Each
    feature then goes through the same fixed pipeline, so that differences in
    score can be attributed to the feature or to the graph alone. Results are
    written after each subject, so a run interrupted midway still leaves usable
    output.

    Parameters
    ----------
    adjacency : {"cov", "cov_beta", "plv_beta", "plv_delta", "plv_alpha"}, default="cov"
        Graph structure used by the convolution. It is computed from the training
        trials of all subjects, saved next to the results, and stays fixed for
        every training run. The five modes are not competing designs but a set of
        controls: "cov" against "cov_beta" holds the measure fixed and varies the
        frequency band, "cov_beta" against "plv_beta" holds the band fixed and
        varies the measure, and the three PLV modes hold the measure fixed while
        moving across bands, delta being the negative control since it carries no
        motor rhythm.

    Returns
    -------
    None
        Results are written to ``OUT_DIR/benchmark_<adjacency>.csv`` with one row
        per subject and feature. Since only the graph differs between runs, the
        CSVs form a paired design and can be compared row by row.

    Examples
    --------
    >>> main()                          # broadband covariance
    >>> main(adjacency="cov_beta")      # beta-band covariance
    >>> main(adjacency="plv_delta")     # delta-band PLV, negative control
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)


    if adjacency == "cov_beta" :

        grand = compute_grand_cov(SUBJECTS, TRAIN_SESSIONS, mode = "beta")   
        path = OUT_DIR / "adj_cov_beta.npz"
        
    elif adjacency == "cov" :

        grand = compute_grand_cov(SUBJECTS, TRAIN_SESSIONS)   
        path = OUT_DIR / "adj_cov.npz"
 
    elif adjacency == "plv_beta" :
        
        grand = compute_grand_plv(SUBJECTS, TRAIN_SESSIONS)  
        path = OUT_DIR / "adj_plv.npz"
    
    elif adjacency == "plv_delta" :
        
        grand = compute_grand_plv(SUBJECTS, TRAIN_SESSIONS, band_idx=0)  
        path = OUT_DIR / "adj_plv_delta.npz"

    elif adjacency == "plv_alpha" :
        
        grand = compute_grand_plv(SUBJECTS, TRAIN_SESSIONS, band_idx=2)  
        path = OUT_DIR / "adj_plv_alpha.npz"

    else : 

        raise ValueError(f"Unknown adjacency type: {adjacency}")
    
    print(f"Computing {adjacency} adjacency...", flush=True)
    np.savez(path, grand_corr=grand)
    adjacency_path = str(path)
    tag = adjacency
    print(f"Adjacency saved to {path}", flush=True)

    results = []

    for subj in SUBJECTS:
        print(f"--- P{subj:02d} ---", flush=True)

        train_trials = TT.get_full_trials(subj, TRAIN_SESSIONS)
        val_trials   = TT.get_full_trials(subj, VAL_SESSIONS)
        test_trials  = TT.get_full_trials(subj, TEST_SESSIONS)
        if not (train_trials and val_trials and test_trials):
            print("  skip: missing sessions", flush=True)
            continue

        train_targets_all = []
        for tr in train_trials:
              
            p = tr['eeg'].shape[0] // N7_PATCH          
            if p < 1:
                continue
            anchors = np.array([(i * N7_PATCH - 1) for i in range(1, p + 1)])
            train_targets_all.append(tr['kin_x'][anchors])   
        train_y = np.concatenate(train_targets_all)
        ym = float(train_y.mean())
        ys = float(train_y.std()+1e-6)
            

            
        #feature extraction 
        f = BandMultiFeatureFrontend(fs=FS, patch_size=N7_PATCH, order="ch_first")
        f_ana = BandAnalyticFrontend(fs=FS, patch_size=N7_PATCH, order="ch_first")
        f0_tr, f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr, f_hand_tr, f_hand_plv_tr, f_bandplv_tr = [], [], [], [], [], [], [], [], [], []
        f0_val, f1_val, f2_val, f3_val, f4_val, f5_val, f6_val, f_hand_val, f_hand_plv_val, f_bandplv_val = [], [], [], [], [], [], [], [], [], []
        f0_te, f1_te, f2_te, f3_te, f4_te, f5_te, f6_te, f_hand_te, f_hand_plv_te, f_bandplv_te = [], [], [], [], [], [], [], [], [], []

        for tr in train_trials:
            
            with torch.no_grad():
                tokens_tr, C_tr, t_tr = f(torch.from_numpy(tr['eeg'][None]).float())
                tokens_ana_tr, C_ana_tr, t_ana_tr = f_ana(torch.from_numpy(tr['eeg'][None]).float())

            handfeat_tr = tokens_tr.reshape(1, C_tr, t_tr, 20).permute(0, 2, 1, 3).contiguous()
            bandpow_tr = tokens_tr[...,5:10].reshape(1, C_tr, t_tr, 5).permute(0, 2, 1, 3).contiguous()


            f1_tr.append(tokens_tr[...,0:5].reshape(1, C_tr, t_tr, 5).permute(0, 2, 1, 3).contiguous()) #linear sample at patch center
            f2_tr.append(bandpow_tr) #log10 mean square (band power)
            f3_tr.append(tokens_tr[...,10:15].reshape(1, C_tr, t_tr, 5).permute(0, 2, 1, 3).contiguous()) #log10 mean absolute first-difference (line length)
            f4_tr.append(tokens_tr[...,15:20].reshape(1, C_tr, t_tr, 5).permute(0, 2, 1, 3).contiguous()) #zero-crossing rate
            f_hand_tr.append(tokens_tr.reshape(1, C_tr, t_tr, 20).permute(0, 2, 1, 3).contiguous())
            plv = compute_plv(tr['eeg'])                         
            plv_agg = plv.mean(axis=3)                            
            plv_feat = np.transpose(plv_agg, (0, 2, 1))           
            plv_feat = torch.from_numpy(plv_feat[None]).float().contiguous()
            f5_tr.append(plv_feat)  

            f6_tr.append(tokens_ana_tr.reshape(1, C_ana_tr, t_ana_tr, 10).permute(0, 2, 1, 3).contiguous())                
            
            handfeat_plv_tr = torch.cat([handfeat_tr, plv_feat], dim=-1)
            f_hand_plv_tr.append(handfeat_plv_tr)
            
            bandpow_plv_tr = torch.cat([bandpow_tr, plv_feat], dim=-1)
            f_bandplv_tr.append(bandpow_plv_tr)

            eeg = tr['eeg']                                    
            t_raw = eeg.shape[0] // N7_PATCH                  
            eeg = eeg[:t_raw * N7_PATCH]                         
            raw = torch.from_numpy(eeg).float()                 
            raw = raw.reshape(t_raw, N7_PATCH, C_tr).permute(0, 2, 1)   
            f0_tr.append(raw[None].contiguous())                   

        for val in val_trials:

            with torch.no_grad():
                tokens_val, C_val, t_val = f(torch.from_numpy(val['eeg'][None]).float())
                tokens_ana_val, C_ana_val, t_ana_val = f_ana(torch.from_numpy(val['eeg'][None]).float())
            
            handfeat_val = tokens_val.reshape(1, C_val, t_val, 20).permute(0, 2, 1, 3).contiguous()
            bandpow_val = tokens_val[...,5:10].reshape(1, C_val, t_val, 5).permute(0, 2, 1, 3).contiguous()


            f1_val.append(tokens_val[...,0:5].reshape(1, C_val, t_val, 5).permute(0, 2, 1, 3).contiguous())
            f2_val.append(bandpow_val)
            f3_val.append(tokens_val[...,10:15].reshape(1, C_val, t_val,5).permute(0, 2, 1, 3).contiguous())
            f4_val.append(tokens_val[...,15:20].reshape(1, C_val, t_val,5).permute(0, 2, 1, 3).contiguous())
            f_hand_val.append(tokens_val.reshape(1, C_val, t_val, 20).permute(0, 2, 1, 3).contiguous())
            plv = compute_plv(val['eeg'])                          # [t, 5, C, C]
            plv_agg = plv.mean(axis=3)                            # [t, 5, C]
            plv_feat = np.transpose(plv_agg, (0, 2, 1))           # [t, C, 5]
            plv_feat = torch.from_numpy(plv_feat[None]).float().contiguous()
            f5_val.append(plv_feat)
            f6_val.append(tokens_ana_val.reshape(1, C_ana_val, t_ana_val, 10).permute(0, 2, 1, 3).contiguous())
            
            handfeat_plv_val = torch.cat([handfeat_val, plv_feat], dim=-1)
            f_hand_plv_val.append(handfeat_plv_val)

            bandpow_plv_val = torch.cat([bandpow_val, plv_feat], dim=-1)
            f_bandplv_val.append(bandpow_plv_val)

            eeg = val['eeg']                                    
            t_raw = eeg.shape[0] // N7_PATCH                    
            eeg = eeg[:t_raw * N7_PATCH]                        
            raw = torch.from_numpy(eeg).float()              
            raw = raw.reshape(t_raw, N7_PATCH, C_val).permute(0, 2, 1)   
            f0_val.append(raw[None].contiguous()) 


        for te in test_trials:

            with torch.no_grad():
                tokens_te, C_te, t_te = f(torch.from_numpy(te['eeg'][None]).float())
                tokens_ana_te, C_ana_te, t_ana_te = f_ana(torch.from_numpy(te['eeg'][None]).float()) 
            
            handfeat_te = tokens_te.reshape(1, C_te, t_te, 20).permute(0, 2, 1, 3).contiguous()
            bandpow_te = tokens_te[...,5:10].reshape(1, C_te, t_te, 5).permute(0, 2, 1, 3).contiguous()


            f1_te.append(tokens_te[...,0:5].reshape(1, C_te, t_te, 5).permute(0, 2, 1, 3).contiguous())
            f2_te.append(bandpow_te)
            f3_te.append(tokens_te[...,10:15].reshape(1, C_te, t_te,5).permute(0, 2, 1, 3).contiguous())
            f4_te.append(tokens_te[...,15:20].reshape(1, C_te, t_te,5).permute(0, 2, 1, 3).contiguous())
            f_hand_te.append(tokens_te.reshape(1, C_te, t_te, 20).permute(0, 2, 1, 3).contiguous())
            plv = compute_plv(te['eeg'])
            plv_agg = plv.mean(axis=3)                            
            plv_feat = np.transpose(plv_agg, (0, 2, 1))           
            plv_feat = torch.from_numpy(plv_feat[None]).float().contiguous()
            f5_te.append(plv_feat)
            f6_te.append(tokens_ana_te.reshape(1, C_ana_te, t_ana_te, 10).permute(0, 2, 1, 3).contiguous())
            
            handfeat_plv_te = torch.cat([handfeat_te, plv_feat], dim=-1)
            f_hand_plv_te.append(handfeat_plv_te)

            bandpow_plv_te = torch.cat([bandpow_te, plv_feat], dim=-1)
            f_bandplv_te.append(bandpow_plv_te)

            eeg = te['eeg']                                    
            t_raw = eeg.shape[0] // N7_PATCH                   
            eeg = eeg[:t_raw * N7_PATCH]                        
            raw = torch.from_numpy(eeg).float()                 
            raw = raw.reshape(t_raw, N7_PATCH, C_te).permute(0, 2, 1)   
            f0_te.append(raw[None].contiguous()) 


        features_tr = [
            ("raw eeg",   f0_tr, 125),
            ("filtered",   f1_tr, 5),
            ("bandpower",  f2_tr, 5),
            ("linelength", f3_tr, 5),
            ("zerocross",  f4_tr, 5),
            ("handfeat",   f_hand_tr, 20),
            ("phase-locking value", f5_tr, 5),
            ("analytic signal", f6_tr, 10),
            ("handfeat + plv", f_hand_plv_tr, 25),
            ("bandpower + plv", f_bandplv_tr, 10),
                ]
        
        features_val = [
            ("raw eeg",   f0_val, 125),
            ("filtered",   f1_val, 5),
            ("bandpower",  f2_val, 5),
            ("linelength", f3_val, 5),
            ("zerocross",  f4_val, 5),
            ("handfeat",   f_hand_val, 20),
            ("phase-locking value", f5_val, 5),
            ("analytic signal", f6_val, 10),
            ("handfeat + plv", f_hand_plv_val, 25),
            ("bandpower + plv", f_bandplv_val, 10),
                ]
        
        features_te = [
            ("raw eeg",   f0_te, 125),
            ("filtered",   f1_te, 5),
            ("bandpower",  f2_te, 5),
            ("linelength", f3_te, 5),
            ("zerocross",  f4_te, 5),
            ("handfeat",   f_hand_te, 20),
            ("phase-locking value", f5_te, 5),
            ("analytic signal", f6_te, 10),
            ("handfeat + plv", f_hand_plv_te, 25),
            ("bandpower + plv", f_bandplv_te, 10),
                ]
     

        for (name,f_tr,fd), (_,f_val,_), (_,f_te,_) in zip(features_tr, features_val, features_te):

            model = Build_Model(PrecomputedFeaturePatchGCN(in_ch=32, feat_dim=fd, embed_dim=8, adj_npz_path=adjacency_path)) #model building
            trained_model = train_streaming_a3(model, f_tr, train_trials, val_trials, f_val, ym, ys)
            r2_mean, r2_med, rmse = eval_stream_a3(trained_model, f_te, test_trials, ym, ys)

            results.append({
                "subject": subj,
                "feature": name,
                "r2_mean": r2_mean,
                "r2_median": r2_med,
                "rmse": rmse
     })
            
        pd.DataFrame(results).to_csv(OUT_DIR / f"benchmark_{tag}.csv", index=False)


if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjacency", default="cov", choices=["cov", "cov_beta", "plv_beta", "plv_delta", "plv_alpha"],
                    help="cov = computed covariance npz ; plv_beta = compute PLV adjacency on the fly on the beta band ; cov_beta =computed covariance (beta band) ; plv_delta = compute PLV adjacency on the fly on the delta band ; plv_alpha = compute PLV adjacency on the fly on the alpha band")
    args = ap.parse_args()
    main(adjacency=(args.adjacency))


import pandas as pd, numpy as np, itertools
from scipy import stats
D="/mnt/project"
ARMS={"cov_orig":"benchmark_cov_ORIG_precomputed_adj_20260806.csv",
      "cov":"benchmark_cov.csv","cov_beta":"benchmark_cov_beta.csv",
      "plv_beta":"benchmark_plv_beta_ORIG_20260806.csv",
      "plv_alpha":"benchmark_plv_alpha.csv","plv_delta":"benchmark_plv_delta.csv",
      "plv_beta_scaled":"benchmark_control_plv_beta_scaled.csv",
      "abs_cov_beta_scaled":"benchmark_control_abs_cov_beta_scaled.csv",
      "uniform_scaled":"benchmark_control_uniform_scaled.csv"}
def load(n): return pd.read_csv(f"{D}/{ARMS[n]}")[["subject","feature","r2_mean"]].rename(columns={"r2_mean":n}).set_index(["subject","feature"])
X=load("cov_orig")
for k in list(ARMS)[1:]: X=X.join(load(k))
print("shape",X.shape,"| any NaN:",X.isna().any().any())
print("\n=== A. arm means ==="); print(X.mean().round(4).to_string())
print("\n=== B. paired contrasts (n=120) ===")
for a,b in [("cov","cov_orig"),("plv_beta","cov"),("plv_beta","cov_orig"),("cov_beta","cov"),
            ("plv_beta","cov_beta"),("plv_delta","plv_beta"),("plv_alpha","plv_beta"),
            ("uniform_scaled","cov_beta"),("uniform_scaled","plv_beta_scaled"),
            ("abs_cov_beta_scaled","plv_beta_scaled"),("plv_beta_scaled","plv_beta")]:
    d=X[a]-X[b]
    print(f"{a:20}-{b:20} mean={d.mean():+.4f} sd={d.std():.3f} p_t={stats.ttest_rel(X[a],X[b]).pvalue:.4f} p_w={stats.wilcoxon(X[a],X[b]).pvalue:.4f} win={100*(d>0).mean():.1f}%")
print("\n=== C. same-graph replicate noise (cov vs cov_orig) ===")
d=X["cov"]-X["cov_orig"]
print(f"n={len(d)} mean={d.mean():+.4f} sd={d.std():.4f} median|d|={d.abs().median():.4f}")
for t in (0.05,0.10,0.20,0.30):
    print(f"  frac |d|>{t:.2f} : {100*(d.abs()>t).mean():.1f}%  ({(d.abs()>t).sum()}/120)")
print("\n=== D. bimodality, cov arm ===")
v=X["cov"]
for lo,hi in [(0,.35),(.35,.70),(.70,1.0)]:
    print(f"  [{lo:.2f},{hi:.2f}) : {100*((v>=lo)&(v<hi)).mean():.1f}%")
print("\n=== E. per-feature, cov arm ===")
g=pd.read_csv(f"{D}/benchmark_cov.csv").groupby("feature").r2_mean.agg(["mean","std","min","max"])
print(g.round(3).sort_values("mean",ascending=False).to_string())
print("\n=== F. feature-rank stability across 9 arms ===")
P=X.groupby("feature").mean()
ks=list(P.columns); rho=[stats.spearmanr(P[a],P[b]).statistic for a,b in itertools.combinations(ks,2)]
print(f"  mean pairwise Spearman = {np.mean(rho):.3f}  min = {np.min(rho):.3f}  max = {np.max(rho):.3f}")
R=P.rank(ascending=False)
print("\n  rank of each feature in each arm (1=best):"); print(R.astype(int).to_string())
print("\n  rank range per feature:"); print((R.max(axis=1)-R.min(axis=1)).astype(int).sort_values().to_string())
print("\n=== G. handfeat per subject, cov vs cov_orig vs plv_beta ===")
h=X.xs("handfeat",level="feature")[["cov_orig","cov","plv_beta"]]
h["cov-cov_orig"]=h["cov"]-h["cov_orig"]
print(h.round(3).to_string()); print("  mean same-graph diff = %+.4f"%h["cov-cov_orig"].mean())

print("\n=== H. collapse rate (cell < 0.35) per feature, pooled over 9 arms ===")
L=X.stack().rename("r2").reset_index(); L.columns=["subject","feature","arm","r2"]
c=L.groupby("feature").r2.apply(lambda s:100*(s<0.35).mean()).sort_values()
print(c.round(1).to_string())
print("\n=== I. collapse rate per subject, pooled ===")
print(L.groupby("subject").r2.apply(lambda s:100*(s<0.35).mean()).round(1).to_string())
print("\n=== J. correlation: feature mean R2 vs its collapse rate (cov arm) ===")
g=pd.read_csv(f"{D}/benchmark_cov.csv").groupby("feature").r2_mean.agg(["mean"])
cc=L[L.arm=="cov"].groupby("feature").r2.apply(lambda s:(s<0.35).mean())
m=g.join(cc.rename("collapse"))
print(m.round(3).to_string())
print("  Pearson r(mean, collapse) = %.3f"%np.corrcoef(m["mean"],m["collapse"])[0,1])
print("\n=== K. conditional mean given NO collapse (r2>=0.35), cov arm ===")
dd=pd.read_csv(f"{D}/benchmark_cov.csv")
k=dd[dd.r2_mean>=0.35].groupby("feature").r2_mean.agg(["mean","count"])
print(k.round(3).sort_values("mean",ascending=False).to_string())
print("\n=== L. adjacency matrices: spread ===")
for f in ["adj_cov_beta.csv","adj_plv_beta.csv","adj_plv_delta.csv","adj_plv_alpha.csv","adj_cov_broadband_recomputed.csv"]:
    a=np.loadtxt(f"{D}/{f}",delimiter=","); off=a[~np.eye(a.shape[0],dtype=bool)]
    print(f"  {f:38} shape={a.shape} off-diag: min={off.min():+.3f} max={off.max():+.3f} mean={off.mean():+.3f} sd={off.sd() if hasattr(off,'sd') else off.std():.3f} frac_neg={100*(off<0).mean():.0f}%")

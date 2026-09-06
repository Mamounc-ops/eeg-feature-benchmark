"""Regenerate all report figures from the benchmark CSVs."""
import pandas as pd, numpy as np, itertools
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/project"; OUT = "figures"
ARMS = {"cov_orig":"benchmark_cov_ORIG_precomputed_adj_20260806.csv",
        "cov":"benchmark_cov.csv","cov_beta":"benchmark_cov_beta.csv",
        "plv_beta":"benchmark_plv_beta_ORIG_20260806.csv",
        "plv_alpha":"benchmark_plv_alpha.csv","plv_delta":"benchmark_plv_delta.csv",
        "plv_beta_scaled":"benchmark_control_plv_beta_scaled.csv",
        "abs_cov_beta_scaled":"benchmark_control_abs_cov_beta_scaled.csv",
        "uniform_scaled":"benchmark_control_uniform_scaled.csv"}
plt.rcParams.update({"font.size":9,"axes.linewidth":.8,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":110})
GREY, DARK = "#D9D9D9", "#333333"

def arm(n):
    return pd.read_csv(f"{DATA}/{ARMS[n]}")[["subject","feature","r2_mean"]]
def wide():
    X = None
    for k in ARMS:
        d = arm(k).rename(columns={"r2_mean":k}).set_index(["subject","feature"])
        X = d if X is None else X.join(d)
    return X
X = wide()
ORDER = arm("cov").groupby("feature").r2_mean.mean().sort_values().index.tolist()
def save(fig,name):
    for e in ("pdf","png"): fig.savefig(f"{OUT}/{name}.{e}",bbox_inches="tight",dpi=200)
    plt.close(fig)

# --- Fig 1 / 2 : feature benchmark under two adjacencies --------------------
def feat_bar(a, name, label, title):
    d = arm(a); m = d.groupby("feature").r2_mean.mean().reindex(ORDER)
    y = np.arange(len(ORDER)); rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4,3.9))
    ax.barh(y, m.values, height=.62, color=GREY, edgecolor="black", lw=.7, zorder=2)
    for i,f in enumerate(ORDER):
        v = d.loc[d.feature==f,"r2_mean"].values
        ax.scatter(v, i+rng.uniform(-.17,.17,v.size), s=9, facecolor="none",
                   edgecolor=DARK, lw=.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.set_xlim(-.02,1)
    ax.set_xticks(np.arange(0,1.01,.2)); ax.set_xlabel(r"$R^2$")
    ax.set_title(f"{label}   {title}", loc="left", fontsize=9.5, pad=7)
    ax.grid(axis="x", color="#EAEAEA", lw=.6); ax.set_axisbelow(True)
    ax.text(.99,.03,"circles = subjects (n=12)",transform=ax.transAxes,
            ha="right",fontsize=7.3,color="#666")
    fig.tight_layout(); save(fig,name)
feat_bar("cov","fig1_features_cov","A","Adjacency: broadband covariance")
feat_bar("plv_beta","fig2_features_plv_beta","B",r"Adjacency: $\beta$-band PLV")

# --- Fig 3 : bimodal cell distribution --------------------------------------
fig,ax = plt.subplots(figsize=(5.4,2.3))
ax.hist(X["cov"].values, bins=np.arange(0,1.01,.05), color=GREY, edgecolor="black", lw=.7)
ax.set_xlabel(r"$R^2$ of one (subject $\times$ feature) cell"); ax.set_ylabel("cells")
ax.set_xlim(0,1); ax.set_title("Cell-level distribution, covariance arm",loc="left",fontsize=9.5,pad=7)
ax.grid(axis="y",color="#EAEAEA",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"fig3_bimodality")

# --- Fig 4 : same graph, two runs (handfeat) --------------------------------
h = X.xs("handfeat",level="feature")[["cov_orig","cov","plv_beta"]]
fig,ax = plt.subplots(figsize=(5.6,3.4))
xs = np.arange(len(h))
ax.plot(xs,h["cov_orig"],"o-",color=DARK,mfc="white",ms=6,lw=1,label="cov (run 1)")
ax.plot(xs,h["cov"],"s-",color=DARK,mfc=DARK,ms=5,lw=1,label="cov (run 2, same matrix)")
ax.plot(xs,h["plv_beta"],"^--",color="#888",mfc="white",ms=6,lw=1,label=r"$\beta$-PLV")
for i in np.where((h["cov"]-h["cov_orig"]).abs()>.3)[0]:
    ax.annotate("",xy=(i,h["cov"].iloc[i]),xytext=(i,h["cov_orig"].iloc[i]),
                arrowprops=dict(arrowstyle="->",color="black",lw=1.1))
ax.set_xticks(xs); ax.set_xticklabels(h.index); ax.set_xlabel("subject")
ax.set_ylabel(r"$R^2$"); ax.set_ylim(0,1)
ax.set_title("handfeat: two runs of the identical covariance graph",loc="left",fontsize=9.5,pad=7)
ax.legend(frameon=False,fontsize=8,loc="lower right")
ax.grid(axis="y",color="#EAEAEA",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"fig4_same_graph_runs")

# --- Fig 5 : replicate-noise distribution -----------------------------------
d = (X["cov"]-X["cov_orig"]).abs()
fig,ax = plt.subplots(figsize=(5.4,2.6))
ax.hist(d.values,bins=np.arange(0,.85,.05),color=GREY,edgecolor="black",lw=.7)
for t,lab in [(.05,"37%"),(.10,"23%"),(.20,"14%")]:
    ax.axvline(t,color="black",lw=.9,ls=":")
    ax.text(t+.012,ax.get_ylim()[1]*.88,f"|Δ|>{t:.2f}\n{lab} of cells",fontsize=7.3,va="top")
ax.set_xlabel(r"$|\Delta R^2|$ between two runs of the identical graph")
ax.set_ylabel("cells"); ax.set_title("Replicate noise floor (n=120 cells)",loc="left",fontsize=9.5,pad=7)
ax.grid(axis="y",color="#EAEAEA",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"fig5_noise_floor")

# --- Fig 6 : arm means against the replicate band ---------------------------
m = X.mean(); rep = abs(X["cov"].mean()-X["cov_orig"].mean())
o = ["cov","plv_alpha","plv_delta","uniform_scaled","plv_beta","abs_cov_beta_scaled",
     "cov_orig","plv_beta_scaled","cov_beta"]
fig,ax = plt.subplots(figsize=(5.6,3.2))
se = X[o].sem()
ax.errorbar(m[o].values,np.arange(len(o)),xerr=se.values,fmt="o",color=DARK,ms=5,
            capsize=2.5,lw=1)
ax.axvspan(m["cov"]-rep,m["cov"]+rep,color="#EAEAEA",zorder=0)
ax.set_yticks(np.arange(len(o))); ax.set_yticklabels(o)
ax.set_xlabel(r"mean $R^2$ over 120 cells")
ax.set_title("Adjacency arms; band = one same-graph replicate difference (0.014)",
             loc="left",fontsize=9,pad=7)
ax.grid(axis="x",color="#EAEAEA",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"fig6_arm_means")

# --- Fig 7 : adjacency matrices ---------------------------------------------
mats = [("adj_cov_broadband_recomputed.csv","covariance (broadband)"),
        ("adj_cov_beta.csv",r"covariance ($\beta$)"),
        ("adj_plv_beta.csv",r"PLV ($\beta$)"),
        ("adj_plv_delta.csv",r"PLV ($\delta$)")]
fig,axes = plt.subplots(1,4,figsize=(9.6,2.8))
for ax,(f,t) in zip(axes,mats):
    a = np.loadtxt(f"{DATA}/{f}",delimiter=",")
    im = ax.imshow(a,vmin=-.3,vmax=1,cmap="gray_r")
    off = a[~np.eye(a.shape[0],dtype=bool)]
    ax.set_title(f"{t}\nsd={off.std():.3f}",fontsize=8.3,pad=5)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im,ax=axes,fraction=.018,pad=.015).set_label("edge weight",fontsize=8)
save(fig,"fig7_adjacency_matrices")

# --- Fig 8 : ranking is a collapse rate ------------------------------------
L = X.stack().rename("r2").reset_index(); L.columns=["subject","feature","arm","r2"]
cr = L.groupby("feature").r2.apply(lambda s:(s<.35).mean())
mm = L.groupby("feature").r2.mean()
fig,ax = plt.subplots(figsize=(4.6,3.4))
ax.scatter(cr*100,mm,s=26,facecolor="none",edgecolor=DARK,lw=.9)
for f in cr.index:
    ax.annotate(f,(cr[f]*100,mm[f]),fontsize=7,xytext=(4,-2),textcoords="offset points")
r = np.corrcoef(cr,mm)[0,1]
ax.set_xlabel("% of cells collapsed ($R^2<0.35$)"); ax.set_ylabel(r"mean $R^2$")
ax.set_title(f"Feature ranking tracks collapse rate (r = {r:.2f})",loc="left",fontsize=9.5,pad=7)
ax.set_xlim(-4,56)
ax.grid(color="#EAEAEA",lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"fig8_collapse_vs_rank")
print("done; r =",round(r,3))

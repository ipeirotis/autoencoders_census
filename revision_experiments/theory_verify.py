# Empirical verification of the measurement model (Fig 1) + Corollary 1 (latent validity >= proxy validity),
# using datasets with MULTIPLE attention checks as conditionally-independent indicators of latent A.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from evaluate.detection import aligned_labels, _relevant, CHECKS
CIT={"racial_data":"ivanov","pennycook_1":"pennycook"}
def scores(ds):
    ae=pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()
    cl=(1-pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    return ae,cl
def base_checks(ds):
    out=[]
    for c in CHECKS[ds]:
        nm=c["name"]
        if "union" in nm.lower() or "inter" in nm.lower(): continue
        out.append(c)
    return out
def auc(y,s): return roc_auc_score(y,np.nan_to_num(s))
for ds in ["racial_data","pennycook_1"]:
    lab=aligned_labels(ds); ae,cl=scores(ds)
    chk=base_checks(ds)
    C=[]; names=[]
    for c in chk:
        v=_relevant(lab,c).astype(float)
        if 0<np.nansum(v)<len(v): C.append(v); names.append(c["name"])
    n=min([len(ae),len(cl)]+[len(v) for v in C])
    ae,cl=ae[:n],cl[:n]; C=[v[:n] for v in C]
    cols=["AE","CL"]+[f"C{i+1}" for i in range(len(C))]
    X=np.column_stack([ae,cl]+C)
    R=np.corrcoef(X.T)
    print(f"\n===== {CIT[ds]}  (n={n}, {len(C)} checks: {names}) =====")
    print("correlation matrix (rows/cols "+", ".join(cols)+"):")
    print(np.round(pd.DataFrame(R,index=cols,columns=cols),2).to_string())
    # --- triad estimate of Corr(M,A) = loading of M on the common factor ---
    idxM={"AE":0,"CL":1}; cidx=list(range(2,2+len(C)))
    for M,mi in idxM.items():
        lams=[]
        for i,j in combinations(cidx,2):
            num=R[mi,i]*R[mi,j]; den=R[i,j]
            if num>0 and den>0: lams.append(np.sqrt(num/den))
        est=np.nanmean(lams) if lams else np.nan
        obs=np.nanmax([R[mi,k] for k in cidx])              # best single-check Pearson (proxy validity)
        aucs=[auc((C[k-2]>0.5).astype(int),X[:,mi]) for k in cidx]
        print(f"  {M}: Corr(M,A)_est={est:.2f} (triads)  vs  best single-check Corr(M,C)={obs:.2f}  "
              f"| single-check AUC range {min(aucs):.2f}-{max(aucs):.2f}")
    # --- 1-factor fit (FactorAnalysis) + RMSR of residual off-diagonal ---
    Z=StandardScaler().fit_transform(np.nan_to_num(X))
    fa=FactorAnalysis(n_components=1,random_state=0).fit(Z)
    W=fa.components_; Rm=W.T@W+np.diag(fa.noise_variance_)
    Remp=np.corrcoef(Z.T); off=~np.eye(len(cols),dtype=bool)
    rmsr=np.sqrt(np.mean((Remp[off]-Rm[off])**2))
    load=W[0]; 
    if np.mean(load[cidx])<0: load=-load                     # orient factor so checks load positive
    print("  1-factor loadings ("+", ".join(cols)+"): "+", ".join(f"{l:.2f}" for l in load))
    print(f"  1-factor RMSR (residual off-diag corr) = {rmsr:.3f}  (small => M and checks conditionally independent given A)")
    # --- composite-check ladder: stricter A-proxy => higher apparent validity (empirical Corollary 1) ---
    fc=np.sum(np.column_stack(C),axis=1)                      # number of checks failed
    print("  AUC(M vs 'fail >= k checks') as the proxy for A gets stricter (cleaner):")
    for M,mi in idxM.items():
        row=[]
        for k in range(1,len(C)+1):
            yk=(fc>=k).astype(int)
            if 0<yk.sum()<len(yk): row.append(f"k>={k}:{auc(yk,X[:,mi]):.2f}")
        print(f"    {M}: "+"  ".join(row))

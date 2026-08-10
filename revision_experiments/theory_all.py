# Measurement-model verification across ALL datasets. Universal: Corollary-1 lower bound on Corr(M,A).
# Where >=2 conditionally-independent checks exist: clean triad Corr(M,A) + single-factor fit + composite ladder.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import aligned_labels, _relevant, CHECKS, _to_ordinal
CIT={"attention_check":"uhalt","inattentive":"alvarez","racial_data":"ivanov","moral_data":"ogrady",
 "mturk_ethics":"moss","bot_bot_mturk":"buchanan","public_opinion":"mastroianni","pennycook_1":"pennycook","sadc_2017":"robinson"}
RHO=0.8   # assumed attention-check reliability rho_max for Corollary 1
def sadc_y():
    dc,rc,ic,adc,arc,aic=define_necessary_elements("sadc_2017",None,None,None)
    L=DataLoader(dc,rc,ic,additional_drop_columns=adc,additional_rename_columns=arc,additional_columns_of_interest=aic)
    return L.find_outlier_data_sadc_2017("sadc_2017",["outlier"])["outlier"].values.astype(int)
def scores(ds):
    if ds=="sadc_2017": ae=pd.read_csv("cache/sadc_2017_85perc_newloss/errors.csv")["error"].to_numpy()
    else: ae=pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()
    cl=(1-pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    return ae,cl
def base_checks(ds):
    if ds=="sadc_2017": return None
    return [c for c in CHECKS[ds] if not any(k in c["name"].lower() for k in ("union","inter"))]
def auc(y,s): return roc_auc_score(y,np.nan_to_num(s))
rows=[]; details=[]
for ds in CIT:
    ae,cl=scores(ds)
    if ds=="sadc_2017":
        y=sadc_y(); C=[y.astype(float)]; names=["outlier"]
    else:
        lab=aligned_labels(ds); C=[]; names=[]
        for c in base_checks(ds):
            v=_relevant(lab,c).astype(float)
            if 0<np.nansum(v)<len(v): C.append(v); names.append(c["name"])
    n=min([len(ae),len(cl)]+[len(v) for v in C]); ae,cl=ae[:n],cl[:n]; C=[v[:n] for v in C]
    prim=(C[0]>0.5).astype(int)
    rMC_ae=abs(np.corrcoef(ae,C[0])[0,1]); rMC_cl=abs(np.corrcoef(cl,C[0])[0,1])
    lb_ae=min(1.0,rMC_ae/RHO); lb_cl=min(1.0,rMC_cl/RHO)
    row=dict(ds=CIT[ds],n=n,checks=len(C),AE_auc=round(auc(prim,ae),2),CL_auc=round(auc(prim,cl),2),
             rMC_CL=round(rMC_cl,2),CorrMA_lb_CL=round(lb_cl,2))
    # multi-check: clean triad Corr(M,A) (M + 2 checks), single-factor RMSR, composite ladder
    if len(C)>=2:
        X=np.column_stack([ae,cl]+C); cols=["AE","CL"]+names; R=np.corrcoef(X.T); cidx=list(range(2,2+len(C)))
        def triad(mi):
            l=[np.sqrt(R[mi,i]*R[mi,j]/R[i,j]) for i,j in combinations(cidx,2) if R[mi,i]*R[mi,j]>0 and R[i,j]>0]
            return np.nanmean(l) if l else np.nan
        cma_ae,cma_cl=triad(0),triad(1)
        Z=StandardScaler().fit_transform(np.nan_to_num(X)); fa=FactorAnalysis(1,random_state=0).fit(Z)
        W=fa.components_; Rm=W.T@W+np.diag(fa.noise_variance_); Remp=np.corrcoef(Z.T); off=~np.eye(len(cols),dtype=bool)
        rmsr=np.sqrt(np.mean((Remp[off]-Rm[off])**2))
        fc=np.sum(np.column_stack(C),1)
        ladder={M:[ (k,round(auc((fc>=k).astype(int),X[:,mi]),2)) for k in range(1,len(C)+1) if 0<(fc>=k).sum()<n]
                 for M,mi in [("AE",0),("CL",1)]}
        row.update(CorrMA_CL=round(cma_cl,2) if not np.isnan(cma_cl) else None,
                   CorrMA_AE=round(cma_ae,2) if not np.isnan(cma_ae) else None, RMSR=round(rmsr,3))
        details.append((CIT[ds],names,ladder,cma_ae,cma_cl))
    rows.append(row)
df=pd.DataFrame(rows); pd.set_option("display.width",200)
print("=== ALL DATASETS: measurement model + Corollary-1 lower bound (rho_max=0.8) ===")
print(df.to_string(index=False))
print("\n rMC_CL = |Corr(CL, primary check)| (observed proxy validity, point-biserial)")
print(" CorrMA_lb_CL = Corollary-1 lower bound on Corr(CL,A) = rMC/rho_max")
print(" CorrMA_CL/AE = CLEAN latent-validity estimate (triad, M + 2 checks); RMSR = single-factor fit (>=2 checks only)")
print("\n=== MULTI-CHECK DETAIL: composite-check AUC ladder (stricter A-proxy => higher validity) ===")
for name,names,ladder,ca,cc in details:
    print(f"  {name} ({len(names)} checks):  triad Corr(M,A) AE={ca:.2f} CL={cc:.2f}")
    for M in ("AE","CL"):
        print(f"     {M}: "+"  ".join(f"fail>={k}:{a}" for k,a in ladder[M]))

# HONEST case-for-our-methods: native-orientation (label-free) evaluation + battery-coherence regime.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import aligned_labels, aligned_battery, _relevant, _to_ordinal, CHECKS
from evaluate import baselines as bl
PRIM={"attention_check":"Attention_Check","inattentive":"filter","racial_data":"attn1","moral_data":"attention",
 "mturk_ethics":"Screener_One","bot_bot_mturk":"Q6_15","public_opinion":"attention_1","pennycook_1":"AC1(screen1)"}
CIT={"attention_check":"uhalt","inattentive":"alvarez","racial_data":"ivanov","moral_data":"ogrady","mturk_ethics":"moss","bot_bot_mturk":"buchanan","public_opinion":"mastroianni","pennycook_1":"pennycook","sadc_2017":"robinson"}
BK=["longstring","irv","person_total_r","mahalanobis","even_odd"]
def sadc():
    dc,rc,ic,adc,arc,aic=define_necessary_elements("sadc_2017",None,None,None)
    L=DataLoader(dc,rc,ic,additional_drop_columns=adc,additional_rename_columns=arc,additional_columns_of_interest=aic)
    y=L.find_outlier_data_sadc_2017("sadc_2017",["outlier"])["outlier"].values.astype(int)
    data,_=L.load_data("sadc_2017"); cols={}
    for c in data.columns:
        e=_to_ordinal(data[c])
        if e is not None and 2<=int(pd.Series(e).dropna().nunique())<=9: cols[c]=np.asarray(e,float)
    return y,pd.DataFrame(cols)
def alpha(bat):
    B=bat.dropna(); k=B.shape[1]
    if k<2: return np.nan
    vi=B.var(0,ddof=1).sum(); vt=B.sum(1).var(ddof=1)
    return float(k/(k-1)*(1-vi/vt)) if vt>0 else np.nan
def load(ds):
    if ds=="sadc_2017":
        y,bat=sadc(); ae=pd.read_csv("cache/sadc_2017_85perc_newloss/errors.csv")["error"].to_numpy()
    else:
        lab=aligned_labels(ds); c=[x for x in CHECKS[ds] if x["name"]==PRIM[ds]][0]; y=_relevant(lab,c).astype(int)
        ae=pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy(); bat=aligned_battery(ds)
    cl=(1-pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    base={k:np.asarray(bl.INDICES[k](bat),float) for k in BK} if bat.shape[1]>=2 else {}
    n=min([len(y),len(ae),len(cl)]+[len(v) for v in base.values()])
    return np.asarray(y)[:n],ae[:n],cl[:n],{k:v[:n] for k,v in base.items()},bat
def nauc(y,s):  # NATIVE orientation: high score = careless, as each detector is designed. No label-driven flip.
    return roc_auc_score(y,np.nan_to_num(s))
rows=[]
for ds in list(PRIM)+["sadc_2017"]:
    y,ae,cl,base,bat=load(ds)
    if not(0<y.sum()<len(y)): continue
    r={"ds":CIT[ds],"alpha":round(alpha(bat),2) if hasattr(bat,'shape') and bat.shape[1]>=2 else np.nan,
       "AE":round(nauc(y,ae),3),"CL":round(nauc(y,cl),3)}
    for k in BK: r[k[:2].upper() if k!="person_total_r" else "PT"]=round(nauc(y,base[k]),3) if k in base else np.nan
    rows.append(r)
df=pd.DataFrame(rows); pd.set_option("display.width",200)
cols=["AE","CL","LO","IR","PT","MA","EV"]
df.columns=[c if c not in ("longstring".upper(),) else c for c in df.columns]
print("=== NATIVE-ORIENTATION AUC (label-free deployment: each index used in its designed direction) ===")
print(df.to_string(index=False))
meth=[c for c in df.columns if c not in ("ds","alpha")]
lead=df[meth].mean().sort_values(ascending=False).round(3)
print("\nMean native AUC (all datasets), ranked:"); print(lead.to_string())
print("\nAnti-detection rate (native AUC < 0.5, i.e. index points the WRONG way and you cannot know without labels):")
anti=(df[meth]<0.5).sum().sort_values()
for m in meth: print(f"  {m}: {int((df[m]<0.5).sum())}/{df[m].notna().sum()} datasets")
# CL/AE best-native count
best_native=df.set_index("ds")[meth].idxmax(axis=1)
print("\nBest NATIVE detector per dataset:"); print(best_native.to_string())
print(f"\nCL or AE is the best native detector on {int(best_native.isin(['AE','CL']).sum())}/{len(best_native)} datasets.")

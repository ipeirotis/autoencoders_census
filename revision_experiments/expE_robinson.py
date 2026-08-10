# Experiment E on Robinson-Cimpian / SADC-2017: does removing OUR unsupervised flag shrink the
# LGBQ-heterosexual drug-use disparity the way removing hand-crafted mischievous responders does?
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re
from utils import define_necessary_elements
from dataset.loader import DataLoader
dc,rc,ic,adc,arc,aic=define_necessary_elements("sadc_2017",None,None,None)
L=DataLoader(dc,rc,ic,additional_drop_columns=adc,additional_rename_columns=arc,additional_columns_of_interest=aic)
misch=L.find_outlier_data_sadc_2017("sadc_2017",["outlier"])["outlier"].values.astype(int)
L.COLUMNS_OF_INTEREST=[]; df,_=L.load_data("sadc_2017"); df=df.reset_index(drop=True)
df=df.loc[:,~df.columns.duplicated()]
ae=pd.read_csv("cache/sadc_2017_85perc_newloss/errors.csv")["error"].to_numpy()
cl=(1-pd.read_csv("cache/_fix9_sadc_2017_cl/errors.csv")["pct"]).to_numpy()
N=len(df); phat=misch.mean()
lgbq=df["sexid"].isin(["Bisexual","Gay or Lesbian","Not Sure"]).to_numpy()
het=(df["sexid"]=="Heterosexual").to_numpy()
DRUGS=["ever_cocaine_use","ever_heroin_use","ever_methamphetamine_use","ever_ecstasy_use",
       "ever_inhalant_use","ever_steroid_use","illegal_injected_drug_use","ever_synthetic_marijuana_use"]
CONTRAST=["considered_suicide","attempted_suicide","weapon_carrying","ever_marijuana_use"]
def binarize(col):
    col=(col.iloc[:,0] if isinstance(col,pd.DataFrame) else col)
    s=col.astype(str)
    no=s.str.contains(r"^0 |never|did not|^no\b|0 times|0 days",case=False,regex=True,na=False)
    y=np.where(col.isna().to_numpy(),np.nan,np.where(no.to_numpy(),0.0,1.0))
    return np.asarray(y,float).ravel()
def keep_topk_removed(score,k): 
    m=np.ones(N,bool); m[np.argsort(-score)[:int(round(k*N))]]=False; return m
def disparity(y,keep):
    y=np.asarray(y,float).ravel(); v=np.isfinite(y)&keep
    a=y[v&lgbq]; b=y[v&het]
    return (a.mean()-b.mean())*100 if len(a)>20 and len(b)>20 else np.nan
conds={"full":np.ones(N,bool),"minus mischievous":misch==0,
       f"minus AE (k=p̂)":keep_topk_removed(ae,phat),f"minus CL (k=p̂)":keep_topk_removed(cl,phat)}
def block(outcomes,title):
    print(f"\n=== {title}: LGBQ-heterosexual disparity (percentage points) ===")
    rows=[]
    for c in outcomes:
        if c not in df.columns: continue
        y=binarize(df[c]); r={"outcome":c.replace('_',' ')}
        for cn,keep in conds.items(): r[cn]=round(disparity(y,keep),1)
        rows.append(r)
    t=pd.DataFrame(rows); print(t.to_string(index=False))
    full=t["full"]; 
    for cn in list(conds)[1:]:
        red=100*(1-(t[cn]/full)); print(f"  mean disparity: full {full.mean():.1f}pp -> {t[cn].mean():.1f}pp  ({cn}: {red.mean():+.0f}% change)")
    return t
pd.set_option("display.width",200)
print(f"N={N}, mischievous p̂={phat:.4f} (n={misch.sum()}); LGBQ={lgbq.sum()}, Het={het.sum()}")
top=lambda s:set(np.argsort(-s)[:misch.sum()]); mset=set(np.where(misch==1)[0])
print(f"overlap of AE-top{misch.sum()} with mischievous: {len(top(ae)&mset)}/{misch.sum()}; CL: {len(top(cl)&mset)}/{misch.sum()}")
td=block(DRUGS,"DRUG OUTCOMES (Robinson-Cimpian predicts inflation)")
tc=block(CONTRAST,"CONTRAST OUTCOMES (literature: suicide/bullying less affected)")
# threshold sweep on the mean drug disparity
print("\n=== threshold sweep k in {0.5,1,1.5,2}*p̂, mean DRUG disparity (pp) ===")
ys=[binarize(df[c]) for c in DRUGS if c in df.columns]
print(f"  full = {np.nanmean([disparity(y,conds['full']) for y in ys]):.1f}")
for mult in [0.5,1,1.5,2]:
    for nm,sc in [("AE",ae),("CL",cl)]:
        keep=keep_topk_removed(sc,mult*phat); md=np.nanmean([disparity(y,keep) for y in ys])
        print(f"  {nm} k={mult}p̂: {md:.1f}", end="   ")
    print()

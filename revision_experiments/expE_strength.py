# Experiment E strength story: does removing OUR unsupervised flag SHARPEN each source paper's headline effect,
# as well as removing attention-check failers does, and better than random removal?
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import aligned_labels, CHECKS, _relevant
def load(ds):
    dc,rc,ic,adc,arc,aic=define_necessary_elements(ds,None,None,None)
    L=DataLoader(dc,rc,ic,additional_drop_columns=adc,additional_rename_columns=arc,additional_columns_of_interest=aic)
    L.COLUMNS_OF_INTEREST=[]; df,_=L.load_data(ds); return df.reset_index(drop=True).loc[:,lambda d:~d.columns.duplicated()]
def scores(ds):
    ae=pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()
    cl=(1-pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy(); return ae,cl
def acfail(ds,n):
    lab=aligned_labels(ds); f=np.zeros(n,bool)
    for ch in CHECKS[ds]: f|=_relevant(lab,ch).astype(bool)
    return f
def num(df,cols): return df[cols].apply(pd.to_numeric,errors="coerce")
def sweep(effect_fn,ae,cl,acf,n,fracs=(0,.05,.10,.20)):
    order={"AE":np.argsort(-ae),"CL":np.argsort(-cl)}; out={}
    for nm,od in order.items():
        out[nm]=[effect_fn(np.setdiff1d(np.arange(n),od[:int(round(f*n))],assume_unique=False)) for f in fracs]
    # random control (avg 20 seeds)
    rnd=[]
    for f in fracs:
        k=int(round(f*n)); vals=[]
        for s in range(20):
            rng=np.random.RandomState(s); rem=rng.choice(n,k,False) if k>0 else np.array([],int)
            vals.append(effect_fn(np.setdiff1d(np.arange(n),rem)))
        rnd.append(np.mean(vals))
    out["random"]=rnd
    out["minus_ACfail"]=effect_fn(np.where(~acf)[0])
    out["n_ACfail"]=int(acf.sum())
    return out
def show(name,out,fracs=(0,.05,.10,.20)):
    print(f"\n### {name}  (effect vs fraction removed) ###")
    for nm in ["AE","CL","random"]:
        print(f"  {nm:7s}: "+"  ".join(f"{int(f*100)}%:{v:.3f}" for f,v in zip(fracs,out[nm])))
    print(f"  minus all AC-failers (n={out['n_ACfail']}): {out['minus_ACfail']:.3f}   [full = {out['AE'][0]:.3f}]")

# ---- pennycook: accuracy discernment = mean(real accurate) - mean(fake accurate) ----
p=load("pennycook_1"); n=len(p); ae,cl=scores("pennycook_1"); acf=acfail("pennycook_1",n)
R=num(p,[c for c in p.columns if c.startswith("Real1_")]).mean(1); F=num(p,[c for c in p.columns if c.startswith("Fake1_")]).mean(1)
def disc(idx): return (R.iloc[idx].mean()-F.iloc[idx].mean())
show("PENNYCOOK accuracy discernment (higher=sharper)",sweep(disc,ae,cl,acf,n))

# ---- alvarez: attitude constraint = mean |inter-item corr| among 6 policy-support items ----
a=load("inattentive"); n=len(a); ae,cl=scores("inattentive"); acf=acfail("inattentive",n)
pol=["support.Obamacare","support.RepealDADT","support.PathLegal","support.CarbonLimits","support.GunControl","support.LimitSurveillance"]
mp={"Support":1.0,"Oppose":-1.0,"Indifferent":0.0,"Don't know":0.0}
P=a[[c for c in pol if c in a.columns]].apply(lambda s:s.map(mp))
def constraint(idx):
    X=P.iloc[idx]; c=X.corr().to_numpy(); iu=np.triu_indices_from(c,1); return np.nanmean(c[iu])
show("ALVAREZ attitude constraint (higher=sharper)",sweep(constraint,ae,cl,acf,n))

# ---- ogrady: MFQ individualizing -> other-regarding behavior (returned the letter) ----
o=load("moral_data"); n=len(o); ae,cl=scores("moral_data"); acf=acfail("moral_data",n)
omap={"bottom-extreme":0,"low":1,"normal":2,"high":3,"top-extreme":4}
indiv=o["individualizing_cat"].map(omap).to_numpy(float)
beh=np.nanmean(np.column_stack([pd.to_numeric(o[b],errors="coerce") for b in ["return1","stamp1","mailed1"]]),axis=1)
def mfqcorr(idx):
    x=indiv[idx]; y=beh[idx]; m=np.isfinite(x)&np.isfinite(y)
    return np.corrcoef(x[m],y[m])[0,1] if m.sum()>20 else np.nan
show("OGRADY MFQ(individualizing)->returned-letter corr (higher=sharper)",sweep(mfqcorr,ae,cl,acf,n))

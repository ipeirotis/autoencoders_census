"""
r2_m5_heldout_p -- referee M5: select the Percentile-Loss p by LEAVE-ONE-DATASET-OUT
so the label-free method never uses the held-out dataset's labels to pick p. Only
p in {85,100} are cached on the current preprocessing (the full p in {80,90,95}
sweep needs AE retraining = TensorFlow, unavailable here -> reported BLOCKED).
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from revision_experiments import lib_r2 as L

P = [85, 100]
auc = {}  # (ds,p) -> AE native AUC on primary check
for ds in L.DS_ALL:
    sc = L.load_scores(ds); y = L.primary_label(ds)
    auc[(ds, 85)] = L.auc_native(y, sc["AE"])
    auc[(ds, 100)] = L.auc_native(y, sc.get("AE_p100", sc["AE"])) if "AE_p100" in sc else np.nan

print(f"{'dataset':14s} {'AE@85':>7s} {'AE@100':>7s}  selected_p(LODO)  heldout_AUC")
sel_aucs, base85 = [], []
for held in L.DS_ALL:
    others = [d for d in L.DS_ALL if d != held and not np.isnan(auc[(d, 100)])]
    # pick p maximizing MEAN AE AUC over the OTHER datasets (no held-out labels used)
    mean_by_p = {p: np.mean([auc[(d, p)] for d in others]) for p in P}
    p_star = max(P, key=lambda p: mean_by_p[p])
    ha = auc[(held, p_star)]
    sel_aucs.append(ha); base85.append(auc[(held, 85)])
    print(f"{L.CITE_ALL[held]:14s} {auc[(held,85)]:7.3f} {auc[(held,100)]:7.3f}   p={p_star:<3d}            {ha:.3f}")

print(f"\nLODO always selects p=85 (detection AUC at 85 > 100 on the training folds).")
print(f"Mean held-out AE AUC (leakage-free selection) = {np.nanmean(sel_aucs):.3f}")
print(f"Mean AE AUC if p were tuned on the eval labels (always-85) = {np.nanmean(base85):.3f}")
print(f"p85 beats p100 on {sum(auc[(d,85)]>auc[(d,100)] for d in L.DS_ALL)}/9 datasets "
      f"(mean gain {np.nanmean([auc[(d,85)]-auc[(d,100)] for d in L.DS_ALL]):+.3f}).")
print("BLOCKED: full p in {80,90,95} held-out sweep needs AE retraining (no TensorFlow); only 85/100 cached.")

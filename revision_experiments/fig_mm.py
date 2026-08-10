import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
OUT="/Users/iliastriantafyllopoulos/Documents/my_projects/inattentiveness_paper/figures"
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":.3,"axes.axisbelow":True,"pdf.fonttype":42})
C={"moss":"#0072B2","pennycook":"#D55E00","ivanov":"#009E73"}
# panel (a): composite-check ladder, Chow-Liu AUC vs number of checks failed
ladder={"moss":[(1,0.73),(2,0.89)],"pennycook":[(1,0.61),(2,0.60),(3,0.54),(4,0.77)],"ivanov":[(1,0.74),(2,0.76)]}
# panel (b): observed proxy validity Corr(M,C) vs estimated latent validity Corr(M,A), Chow-Liu
lat={"ivanov":(0.32,0.40),"moss":(0.24,0.38),"pennycook":(0.13,0.28)}
fig,ax=plt.subplots(1,2,figsize=(9.4,3.9))
for d,pts in ladder.items():
    xs,ys=zip(*pts); ax[0].plot(xs,ys,"-o",color=C[d],lw=1.9,ms=6,label=d)
    ax[0].annotate(f"{ys[-1]:.2f}",(xs[-1],ys[-1]),color=C[d],fontsize=8.5,xytext=(4,-2),textcoords="offset points")
ax[0].set_xlabel("attention-check proxy strictness (number of checks failed)")
ax[0].set_ylabel("Chow-Liu detection AUC"); ax[0].set_xticks([1,2,3,4])
ax[0].set_title("(a) validity rises as the label\napproaches the latent state",fontsize=10.5)
ax[0].legend(frameon=False,fontsize=9,loc="lower right"); ax[0].axhline(0.5,ls=":",color="0.6",lw=1)
ds=list(lat); x=np.arange(len(ds)); w=0.36
ax[1].bar(x-w/2,[lat[d][0] for d in ds],w,color="0.7",label=r"observed $\mathrm{Corr}(M,C)$")
ax[1].bar(x+w/2,[lat[d][1] for d in ds],w,color="#0072B2",label=r"estimated $\mathrm{Corr}(M,A)$")
for i,d in enumerate(ds):
    ax[1].annotate(f"{lat[d][0]:.2f}",(i-w/2,lat[d][0]),ha="center",va="bottom",fontsize=8)
    ax[1].annotate(f"{lat[d][1]:.2f}",(i+w/2,lat[d][1]),ha="center",va="bottom",fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(ds); ax[1].set_ylabel("correlation with attentiveness")
ax[1].set_title("(b) latent validity exceeds\nobserved proxy validity",fontsize=10.5)
ax[1].legend(frameon=False,fontsize=9,loc="upper right"); ax[1].set_ylim(0,0.5)
fig.suptitle("Empirical verification of the measurement model (Chow-Liu detector)",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/measurement_model_verification.pdf"); print("wrote",f"{OUT}/measurement_model_verification.pdf")

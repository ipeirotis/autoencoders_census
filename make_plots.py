# make_plots.py
# Generates scatter plots of Mean AUC vs. dataset characteristics and Mean Lift.

import pandas as pd
import matplotlib.pyplot as plt

# Load the merged summary from correlation_analysis.py
summary = pd.read_csv("out_dataset_summary.csv")

def scatter_plot(x, y, xlab, ylab, out):
    plt.figure()
    plt.scatter(summary[x], summary[y])
    # Least-squares line
    m, b = list(pd.Series(summary[y]).cov(summary[x]) / pd.Series(summary[x]).var() for _ in [0])[0], \
            summary[y].mean() - (summary[x].mean() * list(pd.Series(summary[y]).cov(summary[x]) / pd.Series(summary[x]).var() for _ in [0])[0])
    xs = sorted(summary[x])
    ys = [m*z + b for z in xs]
    plt.plot(xs, ys)
    for i, row in summary.iterrows():
        plt.annotate(row["Dataset"].split(" (")[0], (row[x], row[y]), fontsize=8, alpha=0.8)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

scatter_plot("Samples","MeanAUC","Samples","Mean AUC","auc_vs_samples.png")
scatter_plot("Variables","MeanAUC","# Variables","Mean AUC","auc_vs_variables.png")
scatter_plot("Features","MeanAUC","# One-hot Features","Mean AUC","auc_vs_features.png")
scatter_plot("AFV","MeanAUC","Avg. Features per Variable (AFV)","Mean AUC","auc_vs_afv.png")
scatter_plot("MeanLift","MeanAUC","Mean Lift (reconstruction)","Mean AUC","auc_vs_meanlift.png")

print("Saved plots: auc_vs_*.png")

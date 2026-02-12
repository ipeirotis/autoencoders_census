import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


afv = [3.75, 4.12, 3.65, 3.84, 4.08, 5.03, 4.47, 6.91, 6.51, 6.31, 4.63]
lift = [1.15, 1.50, 1.46, 1.49, 1.53, 1.94, 1.13, 1.95, 1.52, 1.15, 1.65]

# Reshape data for sklearn
X = np.array(afv).reshape(-1, 1)
y = np.array(lift)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate regression line
x_range = np.linspace(min(afv), max(afv), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Create scatter plot with regression line
plt.figure(figsize=(8, 6))
plt.scatter(afv, lift, color='b', edgecolors='k', label="Data points")
plt.plot(x_range, y_pred, color='r', linewidth=2, label="Regression line")

# Labels and title
plt.xlabel("AFV: Average number of Features per Variable", fontsize=17)
plt.ylabel("Lift", fontsize=17)
# plt.title("Scatter plot of AFV vs Lift with Regression Line")

# Show legend and grid
plt.legend(fontsize=17)
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


import seaborn as sns
import pandas as pd

# Create dataframe
data = {
    "Dataset": [
        "Pennycook et al.", "Condition 1", "Condition 2", "Condition 3", "Condition 4",
        "Alvarez et al.", "O et al.", "Buchanan et al.", "Moss et al.", "Mastroianni et al.", "Ivanov et al."
    ],
    "Samples": [853, 212, 206, 220, 215, 2725, 355, 1038, 2277, 1036, 860],
    "Variables": [189, 98, 98, 98, 98, 39, 72, 23, 51, 51, 67],
    "Features": [708, 404, 358, 376, 400, 196, 322, 159, 332, 322, 310],
    "AFV": [3.75, 4.12, 3.65, 3.84, 4.08, 5.03, 4.47, 6.91, 6.51, 6.31, 4.63],
    "Lift": [1.15, 1.50, 1.46, 1.49, 1.53, 1.94, 1.13, 1.95, 1.52, 1.15, 1.65],
}

df = pd.DataFrame(data)

# Pairplot for visualizing correlation of Lift with multiple variables
sns.pairplot(df, vars=["AFV", "Samples", "Variables", "Features", "Lift"], diag_kind="kde", markers="o", plot_kws={"alpha": 0.7})
plt.show()

# Create a figure for multiple regression plots
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

# Define variables to compare with Lift
variables = ["AFV", "Variables", "Features", "Samples"]
titles = ["AFV vs Lift", "# of Variables vs Lift", "# of Features vs Lift", "# of Samples vs Lift"]

for i, var in enumerate(variables):
    X = df[[var]]
    y = df["Lift"]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate regression line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Scatter plot with regression line
    axes[i].scatter(df[var], df["Lift"], color='b', edgecolors='k', alpha=0.6, label="Data points")
    axes[i].plot(x_range, y_pred, color='r', linewidth=2, label="Regression line")

    axes[i].set_xlabel(var, fontsize=15)
    axes[i].set_title(titles[i], fontsize=15)
    axes[i].grid(True, linestyle='--', alpha=0.6)

# Set y-axis label for the first plot only
axes[0].set_ylabel("Lift", fontsize=15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14, frameon=True)

from scipy.stats import pearsonr

# Compute Pearson correlation coefficients
correlations = {var: pearsonr(df[var], df["Lift"])[0] for var in ["AFV", "Variables", "Features", "Samples"]}

# Display correlation coefficients
correlation_df = pd.DataFrame(list(correlations.items()), columns=["Variable", "Pearson Correlation with Lift"])

print(correlation_df)

"""Week 11: Data Analysis - Exploratory Data Analysis on the Iris dataset."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Load the data
df = pd.read_csv("Iris.csv")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

# 2. Basic inspection
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows (ignoring Id):", df.drop(columns="Id").duplicated().sum())
print("\nClass counts:\n", df["Species"].value_counts())

# 3. Summary statistics
print("\nOverall summary statistics:\n", df[features].describe().round(3))
print("\nMean by species:\n", df.groupby("Species")[features].mean().round(3))
print("\nMedian by species:\n", df.groupby("Species")[features].median().round(3))
print("\nStd dev by species:\n", df.groupby("Species")[features].std().round(3))
print("\nSkewness:\n", df[features].skew().round(3))
print("\nKurtosis:\n", df[features].kurt().round(3))
print("\nCorrelation matrix:\n", df[features].corr().round(3))

# 4. Visualizations
# 4a. Histograms
df[features].hist(bins=15, figsize=(10, 7), edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("1_histograms.png", dpi=150)
plt.close()

# 4b. Box plots by species
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="Species", y=col, hue="Species", ax=ax, legend=False)
    ax.set_title(col)
plt.suptitle("Box Plots by Species")
plt.tight_layout()
plt.savefig("2_boxplots.png", dpi=150)
plt.close()

# 4c. Violin plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), features):
    sns.violinplot(data=df, x="Species", y=col, hue="Species", ax=ax, legend=False)
    ax.set_title(col)
plt.suptitle("Violin Plots by Species")
plt.tight_layout()
plt.savefig("3_violinplots.png", dpi=150)
plt.close()

# 4d. Pair plot
g = sns.pairplot(df, vars=features, hue="Species", diag_kind="kde")
g.figure.suptitle("Pair Plot", y=1.02)
g.savefig("4_pairplot.png", dpi=120)
plt.close("all")

# 4e. Scatter plots (petal and sepal)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", ax=axes[0])
axes[0].set_title("Petal Length vs Width")
sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=axes[1])
axes[1].set_title("Sepal Length vs Width")
plt.tight_layout()
plt.savefig("5_scatter.png", dpi=150)
plt.close()

# 4f. Correlation heatmap
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("6_heatmap.png", dpi=150)
plt.close()

# 4g. Species count
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Species", hue="Species", legend=False)
plt.title("Samples per Species")
plt.tight_layout()
plt.savefig("7_countplot.png", dpi=150)
plt.close()

print("\nAll plots saved.")

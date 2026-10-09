"""Week 12: Decision tree classifier on the Iris dataset."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay)

# 1. Load data
df = pd.read_csv("Iris.csv")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X, y = df[features], df["Species"]

# 2. Train/test split (80/20, stratified so each species is equally represented)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Train the decision tree
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro)   : {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall    (macro)   : {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall    (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print("\nPer-class report:\n", classification_report(y_test, y_pred, digits=3))

# 5. Feature importance
imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("Feature importances:\n", imp.round(3))

# 6. Plots
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("dt_confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(16, 9))
plot_tree(clf, feature_names=features, class_names=clf.classes_,
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("dt_tree.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
imp.sort_values().plot(kind="barh")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("dt_feature_importance.png", dpi=150)
plt.close()

# 7. Optional: 5-fold cross-validation for a more reliable estimate
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5)
print(f"\n5-fold CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

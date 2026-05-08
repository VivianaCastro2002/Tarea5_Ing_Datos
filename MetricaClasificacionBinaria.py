import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
plt.style.use("default")

digits = load_digits()
X = digits.data
y_multiclase = digits.target
y_es_5 = y_multiclase == 5
imagenes = digits.images

print(f"Observaciones: {X.shape[0]:,}")
print(f"Variables por imagen: {X.shape[1]}")
print(f"Cantidad de imágenes que son 5: {y_es_5.sum():,}")
print(f"Proporción de clase positiva: {y_es_5.mean():.3f}")

fig, axes = plt.subplots(2, 5, figsize=(9, 4))
indices_5 = np.where(y_es_5)[0][:5]
indices_no_5 = np.where(~y_es_5)[0][:5]
indices = np.concatenate([indices_5, indices_no_5])

for indice, ax in zip(indices, axes.ravel()):
    ax.imshow(imagenes[indice], cmap="gray_r")
    etiqueta = "5" if y_es_5[indice] else "no 5"
    ax.set_title(etiqueta)
    ax.axis("off")

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test, imagenes_train, imagenes_test = train_test_split(
    X,
    y_es_5,
    imagenes,
    test_size=0.25,
    stratify=y_es_5,
    random_state=RANDOM_STATE,
)

resumen = pd.DataFrame({
    "filas": [len(X_train), len(X_test)],
    "proporción_positiva": [y_train.mean(), y_test.mean()],
}, index=["train", "test"])

display(resumen)

clasificador_sgd = Pipeline(steps=[
    ("escalamiento", StandardScaler()),
    ("modelo", SGDClassifier(random_state=RANDOM_STATE)),
])

clasificador_sgd.fit(X_train, y_train)
y_pred_sgd = clasificador_sgd.predict(X_test)
scores_sgd = clasificador_sgd.decision_function(X_test)

print(f"Primeros scores: {np.round(scores_sgd[:10], 2)}")
print(f"Primeras predicciones: {y_pred_sgd[:10]}")

cm = confusion_matrix(y_test, y_pred_sgd)
cm_df = pd.DataFrame(cm, index=["real_no_5", "real_5"], columns=["pred_no_5", "pred_5"])
display(cm_df)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_sgd,
    display_labels=["no 5", "5"],
    cmap="Blues",
    values_format="d",
    ax=ax,
)
ax.set_title("Matriz de confusión - detector de 5")
plt.tight_layout()
plt.show()


tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

metricas_sgd = pd.DataFrame([{
    "accuracy": accuracy_score(y_test, y_pred_sgd),
    "precision": precision_score(y_test, y_pred_sgd, zero_division=0),
    "recall": recall_score(y_test, y_pred_sgd, zero_division=0),
    "f1": f1_score(y_test, y_pred_sgd, zero_division=0),
}], index=["SGDClassifier"])

display(metricas_sgd.style.format("{:.3f}"))
print(classification_report(y_test, y_pred_sgd, target_names=["no 5", "5"], zero_division=0))

def metricas_para_umbral(threshold):
    y_pred_threshold = scores_sgd >= threshold
    return {
        "threshold": threshold,
        "positivos_predichos": int(y_pred_threshold.sum()),
        "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
        "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
        "f1": f1_score(y_test, y_pred_threshold, zero_division=0),
    }

thresholds_a_probar = [-5, 0, 5, 10]
tabla_thresholds = pd.DataFrame([metricas_para_umbral(t) for t in thresholds_a_probar])
display(tabla_thresholds.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}))


precision, recall, thresholds = precision_recall_curve(y_test, scores_sgd)

precisions, recalls, pr_thresholds = precision_recall_curve(y_test, scores_sgd)
average_precision = average_precision_score(y_test, scores_sgd)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(recalls, precisions, color="#4C78A8")
ax.set_title(f"Curva precision-recall (AP={average_precision:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisión")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

precision_threshold = precisions[:-1]
recall_threshold = recalls[:-1]
f1_threshold = 2 * precision_threshold * recall_threshold / (precision_threshold + recall_threshold + 1e-12)
mejor_indice = int(np.nanargmax(f1_threshold))
threshold_f1 = float(pr_thresholds[mejor_indice])

print(f"Threshold que maximiza F1 en test: {threshold_f1:.3f}")
print(f"Precisión: {precision_threshold[mejor_indice]:.3f}")
print(f"Recall: {recall_threshold[mejor_indice]:.3f}")
print(f"F1: {f1_threshold[mejor_indice]:.3f}")

fpr_sgd, tpr_sgd, roc_thresholds = roc_curve(y_test, scores_sgd)
auc_sgd = roc_auc_score(y_test, scores_sgd)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr_sgd, tpr_sgd, label=f"SGDClassifier (AUC={auc_sgd:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="azar")
ax.set_title("Curva ROC")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate / Recall")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
scores_rf = random_forest.predict_proba(X_test)[:, 1]

comparacion = pd.DataFrame([
    {
        "modelo": "SGDClassifier",
        "accuracy": accuracy_score(y_test, y_pred_sgd),
        "precision": precision_score(y_test, y_pred_sgd, zero_division=0),
        "recall": recall_score(y_test, y_pred_sgd, zero_division=0),
        "f1": f1_score(y_test, y_pred_sgd, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores_sgd),
        "average_precision": average_precision_score(y_test, scores_sgd),
    },
    {
        "modelo": "RandomForestClassifier",
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores_rf),
        "average_precision": average_precision_score(y_test, scores_rf),
    },
]).set_index("modelo")

display(comparacion.style.format("{:.3f}"))

fpr_rf, tpr_rf, _ = roc_curve(y_test, scores_rf)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr_sgd, tpr_sgd, label=f"SGD (AUC={auc_sgd:.3f})")
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, scores_rf):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="azar")
ax.set_title("Comparación de curvas ROC")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate / Recall")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


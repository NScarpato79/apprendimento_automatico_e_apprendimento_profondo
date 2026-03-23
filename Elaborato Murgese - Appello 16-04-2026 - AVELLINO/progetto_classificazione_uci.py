# -*- coding: utf-8 -*-
"""
Progetto di Classificazione su Dataset UCI
=========================================

Questo script Python implementa un esperimento completo di classificazione su un
**dataset UCI** (di default: *Iris*), comprendente:

1) Analisi PCA con visualizzazioni.
2) Almeno tre algoritmi di classificazione (Logistic Regression, SVM, Random Forest, KNN).
3) Calcolo delle metriche: Accuracy, Precision, Recall, F1, ROC AUC (macro/micro ove applicabile).
4) Visualizzazione matrice di confusione e curve ROC AUC.
5) Opzione di *deep learning* (Keras/TensorFlow se disponibile; fallback a MLPClassifier di scikit-learn)
   utilizzando **la stessa suddivisione** train/val/test.

Uso (esempi):
-------------

    # Esecuzione con dataset UCI Iris (default)
    python progetto_classificazione_uci.py

    # Esecuzione con un CSV locale (specifica colonna target)
    python progetto_classificazione_uci.py --dataset csv --csv_path ./nomedataset.csv --target_col target

Dipendenze principali:
----------------------
- numpy, pandas, matplotlib, seaborn, scikit-learn
- (Opz.) tensorflow/keras per la parte deep learning (altrimenti fallback automatico a sklearn MLP)

Output:
-------
- Cartella `outputs/<timestamp>/` con figure (PCA, confusion matrix, ROC) e `report.md` riepilogativo.

Nota:
-----
Questo script è auto-contenuto su dataset UCI forniti da scikit-learn (Iris/Wine/Breast Cancer). 
Per dataset personalizzati in CSV, specificare `--dataset csv`, `--csv_path` e `--target_col`.

Autore: Daniele Murgese
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


def ensure_outdir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_dataset(dataset: str, csv_path: str = None, target_col: str = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], str]:
    dataset = dataset.lower()

    if dataset == "iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        target_names = list(data.target_names)
        feature_names = list(data.feature_names)
        name = "Iris (UCI)"
    elif dataset == "wine":
        data = datasets.load_wine()
        X = data.data
        y = data.target
        target_names = list(data.target_names)
        feature_names = list(data.feature_names)
        name = "Wine (UCI)"
    elif dataset == "breast_cancer":
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        target_names = list(data.target_names)
        feature_names = list(data.feature_names)
        name = "Breast Cancer Wisconsin (UCI)"
    elif dataset == "csv":
        if csv_path is None or target_col is None:
            raise ValueError("Per dataset=csv specificare --csv_path e --target_col")
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"Colonna target '{target_col}' non trovata nel CSV")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = [c for c in df.columns if c != target_col]
        if pd.api.types.is_numeric_dtype(df[target_col]):
            classes_sorted = sorted(np.unique(y))
            target_names = [str(c) for c in classes_sorted]
            class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
            y = np.array([class_to_idx[val] for val in y])
        else:
            classes_sorted = sorted(df[target_col].unique())
            class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
            target_names = [str(c) for c in classes_sorted]
            y = np.array([class_to_idx[val] for val in y])
        name = f"CSV personalizzato ({os.path.basename(csv_path)})"
    else:
        raise ValueError("dataset non supportato. Usa: iris | wine | breast_cancer | csv")

    return X, y, target_names, feature_names, name


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_rel, random_state=random_state, stratify=y_train_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def perform_pca(
    X_train_s: np.ndarray,
    X_val_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: np.ndarray,
    target_names: List[str],
    out_dir: str,
    max_components: int = 10,
) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    n_features = X_train_s.shape[1]
    n_comp = min(max_components, n_features)
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X_train_s)

    X_train_p = pca.transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_comp + 1), evr, alpha=0.7, label='Varianza spiegata')
    plt.step(range(1, n_comp + 1), cum_evr, where='mid', color='red', label='Cumulata')
    plt.xlabel('Componente Principale')
    plt.ylabel('Quota di Varianza Spiegata')
    plt.title('PCA: Varianza spiegata per componente')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_varianza_spiegata.png'), dpi=150)
    plt.close()

    if X_train_p.shape[1] >= 2:
        plt.figure(figsize=(7, 6))
        palette = sns.color_palette('tab10', n_colors=len(np.unique(y_train)))
        for cls in np.unique(y_train):
            idx = y_train == cls
            plt.scatter(
                X_train_p[idx, 0], X_train_p[idx, 1],
                s=35, alpha=0.8, label=target_names[cls] if cls < len(target_names) else str(cls),
                color=palette[int(cls) % len(palette)]
            )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: Proiezione 2D (train)')
        plt.legend(title='Classi')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pca_scatter2d_train.png'), dpi=150)
        plt.close()

    return pca, X_train_p, X_val_p, X_test_p


def get_classifiers(random_state: int) -> Dict[str, object]:
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=200, n_jobs=None, random_state=random_state),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    return classifiers


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(y_test_bin: np.ndarray, y_score: np.ndarray, class_names: List[str], title_prefix: str, out_path: str):
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr['micro'], tpr['micro'], label=f"micro-average ROC (AUC = {roc_auc['micro']:.3f})", color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr['macro'], tpr['macro'], label=f"macro-average ROC (AUC = {roc_auc['macro']:.3f})", color='navy', linestyle=':', linewidth=3)

    colors = sns.color_palette('tab10', n_colors=n_classes)
    for i, color in enumerate(colors):
        nm = class_names[i] if i < len(class_names) else str(i)
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f"ROC classe {nm} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falso Positivo (FPR)')
    plt.ylabel('Vero Positivo (TPR)')
    plt.title(f'{title_prefix} - Curve ROC (OvR)')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: str,
    model_name: str,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X_test)
        if score.ndim == 1:
            score = np.vstack([1 - score, score]).T
        y_score = score
    else:
        n_classes = len(np.unique(y_test))
        y_score = np.zeros((len(y_pred), n_classes))
        for i, c in enumerate(y_pred):
            y_score[i, c] = 1.0

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    classes_sorted = np.unique(y_test)
    n_classes = len(classes_sorted)
    y_test_bin = label_binarize(y_test, classes=classes_sorted)

    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_score[:, classes_sorted], average='macro', multi_class='ovr') if n_classes > 1 else np.nan
        roc_auc_micro = roc_auc_score(y_test_bin, y_score[:, classes_sorted], average='micro', multi_class='ovr') if n_classes > 1 else np.nan
    except Exception:
        roc_auc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr') if n_classes > 1 else np.nan
        roc_auc_micro = roc_auc_score(y_test_bin, y_score, average='micro', multi_class='ovr') if n_classes > 1 else np.nan

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(out_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, f"{model_name} - Matrice di Confusione (test)", cm_path)

    if n_classes > 1:
        try:
            y_score_for_roc = y_score[:, classes_sorted]
        except Exception:
            y_score_for_roc = y_score
        roc_path = os.path.join(out_dir, f"{model_name.replace(' ', '_').lower()}_roc_curves.png")
        plot_roc_curves(y_test_bin, y_score_for_roc, class_names, model_name, roc_path)

    metrics = {
        'accuracy': float(acc),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'roc_auc_macro': float(roc_auc_macro) if not np.isnan(roc_auc_macro) else None,
        'roc_auc_micro': float(roc_auc_micro) if not np.isnan(roc_auc_micro) else None,
    }
    return metrics


def train_deep_learning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: str,
    random_state: int,
) -> Tuple[str, Dict[str, float]]:
    n_classes = len(np.unique(y_train))

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.random.set_seed(random_state)

        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            verbose=0,
            callbacks=callbacks
        )

        hist = history.history
        plt.figure(figsize=(8, 4))
        plt.plot(hist['accuracy'], label='Train Acc')
        plt.plot(hist['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoca')
        plt.ylabel('Accuratezza')
        plt.title('Rete DNN - Curve di Apprendimento')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'dnn_learning_curves.png'), dpi=150)
        plt.close()

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        try:
            roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr') if n_classes > 1 else np.nan
            roc_auc_micro = roc_auc_score(y_test_bin, y_prob, average='micro', multi_class='ovr') if n_classes > 1 else np.nan
        except Exception:
            roc_auc_macro, roc_auc_micro = np.nan, np.nan

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, class_names, 'DNN - Matrice di Confusione (test)', os.path.join(out_dir, 'dnn_confusion_matrix.png'))

        if n_classes > 1:
            plot_roc_curves(y_test_bin, y_prob, class_names, 'DNN', os.path.join(out_dir, 'dnn_roc_curves.png'))

        metrics = {
            'accuracy': float(acc),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'roc_auc_macro': float(roc_auc_macro) if not np.isnan(roc_auc_macro) else None,
            'roc_auc_micro': float(roc_auc_micro) if not np.isnan(roc_auc_micro) else None,
        }
        return 'Deep NN (Keras/TensorFlow)', metrics

    except Exception as e:
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500, random_state=random_state)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, class_names, out_dir, 'MLP (sklearn)')
        return 'MLP (sklearn fallback)', metrics


def write_report(
    out_dir: str,
    dataset_name: str,
    n_samples: int,
    n_features: int,
    class_names: List[str],
    pca_evr: np.ndarray,
    metrics_dict: Dict[str, Dict[str, float]],
):
    lines = []
    lines.append(f"# Report Esperimento di Classificazione\n")
    lines.append(f"**Dataset:** {dataset_name}\n")
    lines.append(f"**Campioni:** {n_samples} | **Feature:** {n_features} | **Classi:** {len(class_names)} -> {', '.join(map(str, class_names))}\n")
    lines.append("\n## PCA\n")
    lines.append("Varianza spiegata (prime componenti):\n")
    lines.append(", ".join([f"PC{i+1}:{v:.3f}" for i, v in enumerate(pca_evr)]) + "\n")
    lines.append("Figure: `pca_varianza_spiegata.png`, `pca_scatter2d_train.png`\n")

    lines.append("\n## Metriche di Valutazione (test)\n")
    header = ["Modello", "Accuracy", "Precision (w)", "Recall (w)", "F1 (w)", "ROC AUC (macro)", "ROC AUC (micro)"]
    lines.append("| " + " | ".join(header) + " |\n")
    lines.append("|" + "---|" * (len(header)) + "\n")

    for model_name, m in metrics_dict.items():
        row = [
            model_name,
            f"{m.get('accuracy', np.nan):.4f}" if m.get('accuracy') is not None else "-",
            f"{m.get('precision_weighted', np.nan):.4f}" if m.get('precision_weighted') is not None else "-",
            f"{m.get('recall_weighted', np.nan):.4f}" if m.get('recall_weighted') is not None else "-",
            f"{m.get('f1_weighted', np.nan):.4f}" if m.get('f1_weighted') is not None else "-",
            f"{m.get('roc_auc_macro', np.nan):.4f}" if m.get('roc_auc_macro') is not None else "-",
            f"{m.get('roc_auc_micro', np.nan):.4f}" if m.get('roc_auc_micro') is not None else "-",
        ]
        lines.append("| " + " | ".join(row) + " |\n")

    report_path = os.path.join(out_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description='Progetto di classificazione su dataset UCI con PCA, modelli classici, metriche e deep learning.')
    parser.add_argument('--dataset', type=str, default='iris', help='iris | wine | breast_cancer | csv')
    parser.add_argument('--csv_path', type=str, default=None, help='Percorso al CSV (se --dataset csv)')
    parser.add_argument('--target_col', type=str, default=None, help='Nome della colonna target (se --dataset csv)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Frazione test')
    parser.add_argument('--val_size', type=float, default=0.2, help='Frazione validation (dell\'intero dataset)')
    parser.add_argument('--random_state', type=int, default=42, help='Seed')
    parser.add_argument('--no_scale', action='store_true', help='Non applicare StandardScaler (sconsigliato)')

    args = parser.parse_args()

    out_dir = ensure_outdir()

    X, y, target_names, feature_names, dataset_name = load_dataset(args.dataset, args.csv_path, args.target_col)
    n_samples, n_features = X.shape

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state
    )

    if args.no_scale:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
        scaler = None
    else:
        X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    pca, X_train_p, X_val_p, X_test_p = perform_pca(
        X_train_s, X_val_s, X_test_s, y_train, target_names, out_dir
    )

    clfs = get_classifiers(args.random_state)
    metrics_dict = {}
    for name, clf in clfs.items():
        clf.fit(X_train_s, y_train)
        m = evaluate_model(clf, X_test_s, y_test, target_names, out_dir, name)
        metrics_dict[name] = m

    dl_name, dl_metrics = train_deep_learning(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, target_names, out_dir, args.random_state
    )
    metrics_dict[dl_name] = dl_metrics

    write_report(
        out_dir=out_dir,
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        class_names=target_names,
        pca_evr=pca.explained_variance_ratio_,
        metrics_dict=metrics_dict,
    )

    print("\n=== Esperimento completato ===")
    print(f"Output: {out_dir}")
    print(f"Dataset: {dataset_name} | Campioni: {n_samples} | Feature: {n_features} | Classi: {len(target_names)}")
    print("\nMetriche (test):")
    for k, v in metrics_dict.items():
        print(f"- {k}: Acc={v.get('accuracy'):.4f}, Prec(w)={v.get('precision_weighted'):.4f}, Recall(w)={v.get('recall_weighted'):.4f}, F1(w)={v.get('f1_weighted'):.4f}, ROC-AUC(macro)={v.get('roc_auc_macro')} ")


if __name__ == '__main__':
    main()

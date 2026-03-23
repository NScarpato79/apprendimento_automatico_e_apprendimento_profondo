# Report Esperimento di Classificazione
**Dataset:** Iris (UCI)
**Campioni:** 150 | **Feature:** 4 | **Classi:** 3 -> setosa, versicolor, virginica

## PCA
Varianza spiegata (prime componenti):
PC1:0.724, PC2:0.238, PC3:0.034, PC4:0.005
Figure: `pca_varianza_spiegata.png`, `pca_scatter2d_train.png`

## Metriche di Valutazione (test)
| Modello | Accuracy | Precision (w) | Recall (w) | F1 (w) | ROC AUC (macro) | ROC AUC (micro) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9917 | 0.9950 |
| SVM (RBF) | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9967 | 0.9983 |
| Random Forest | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9833 | 0.9822 |
| KNN | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9983 | 0.9983 |
| MLP (sklearn fallback) | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9950 | 0.9961 |

# Progetto — Apprendimento Automatico e Apprendimento Profondo

Esperimento di **classificazione supervisionata** sul dataset UCI
**Breast Cancer Wisconsin (Diagnostic)** (569 campioni, 30 feature, 2 classi: *malignant* / *benign*).

**Autore:** Andrea Balillo

## Contenuto del progetto

Il progetto implementa tutti i task richiesti dalla traccia d'esame:

| Task | Descrizione | Implementazione |
|------|-------------|-----------------|
| 1 | Analisi delle feature con **PCA** e visualizzazione | scree plot, varianza cumulata, proiezione 2D, loadings |
| 2 | Almeno tre algoritmi di classificazione | **5 modelli**: Logistic Regression, SVM (RBF), Random Forest, KNN, Naive Bayes |
| 3 | Metriche: precision, recall, f-measure, accuracy, ROC AUC | calcolate su tutti i modelli |
| 4 | Visualizzazione matrice di confusione e curva ROC | grafici prodotti in Python |
| 5a | Approccio di **deep learning** (stesso split) | rete neurale **MLP in PyTorch** |
| 5b | Approccio di **spiegabilità** | **SHAP** (beeswarm, bar, waterfall) |

> Sono state realizzate **entrambe** le opzioni del Task 5 (a *e* b), pur essendone richiesta una sola.

## Struttura dei file

```
.
├── notebook_progetto.ipynb      # Notebook principale (eseguibile end-to-end) — DELIVERABLE
├── Relazione_Progetto_AAAP.pdf  # Relazione in PDF — DELIVERABLE
├── progetto_src.py              # Versione .py del codice del notebook
├── requirements.txt             # Dipendenze
├── results.json                 # Metriche esportate (usate per la relazione)
├── figures/                     # Tutte le figure generate (PNG)
└── README.md
```

## Come eseguire

1. (Consigliato) creare un ambiente virtuale:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
2. Installare le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
3. Aprire ed eseguire il notebook:
   ```bash
   jupyter notebook notebook_progetto.ipynb
   ```
   oppure eseguirlo interamente da riga di comando:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebook_progetto.ipynb
   ```

Il dataset viene caricato automaticamente da `scikit-learn` (`load_breast_cancer`), copia fedele del
dataset UCI: **non è necessaria alcuna connessione** né download manuale. Tutti gli esperimenti usano
un *random seed* fisso (42) per la piena riproducibilità.

## Riepilogo dei risultati (test set)

| Modello | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | 0.9912 | **0.9971** |
| SVM (RBF) | 0.9649 | 0.9961 |
| MLP (Deep Learning) | 0.9825 | 0.9941 |
| Random Forest | 0.9298 | 0.9902 |
| Naive Bayes | 0.9123 | 0.9859 |
| K-Nearest Neighbors | 0.9474 | 0.9812 |

Il modello migliore è la **Logistic Regression** (ROC AUC 0.997), con la rete neurale su prestazioni
comparabili — coerentemente con l'ottima separabilità lineare evidenziata dalla PCA.

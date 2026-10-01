# Progetto — Apprendimento Automatico e Apprendimento Profondo

Esperimento di **classificazione supervisionata multiclasse** sul dataset UCI
**Dry Bean** (13.611 campioni, 16 feature morfologiche, **7 classi** di fagioli secchi:
*BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA*).

**Autore:** Andrea Vittorio Balillo — **Matricola:** 0322500157

## Contenuto del progetto

Il progetto implementa tutti i task richiesti dalla traccia d'esame:

| Task | Descrizione | Implementazione |
|------|-------------|-----------------|
| 1 | Analisi delle feature con **PCA** e visualizzazione | scree plot, varianza cumulata, proiezione 2D, loadings |
| 2 | Almeno tre algoritmi di classificazione | **5 modelli**: Logistic Regression, SVM (RBF), Random Forest, KNN, Naive Bayes |
| 3 | Metriche: precision, recall, f-measure, accuracy, ROC AUC | calcolate su tutti i modelli (macro-average / One-vs-Rest) |
| 4 | Visualizzazione matrice di confusione e curva ROC | matrici 7×7 e curve ROC macro-average |
| 5a | Approccio di **deep learning** (stesso split) | rete neurale **MLP in PyTorch** (16→64→32→7) |
| 5b | Approccio di **spiegabilità** | **SHAP** (bar, beeswarm, waterfall) |

> Sono state realizzate **entrambe** le opzioni del Task 5 (a *e* b), pur essendone richiesta una sola.

## Struttura dei file

```
a_balillo_0322500157/
├── notebook_a_balillo_0322500157.ipynb   # Notebook principale (eseguibile end-to-end) — DELIVERABLE
├── relazione_a_balillo_0322500157.pdf    # Relazione in PDF — DELIVERABLE
├── dry_bean.csv                          # Dataset (per esecuzione offline)
├── progetto_src.py                       # Versione .py del codice del notebook
├── requirements.txt                      # Dipendenze
├── results.json                          # Metriche esportate (usate per la relazione)
├── figures/                              # Tutte le figure generate (14 PNG)
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
   jupyter notebook notebook_a_balillo_0322500157.ipynb
   ```

Il dataset viene letto dal file locale **`dry_bean.csv`**: **non è necessaria alcuna
connessione** né download manuale. Se il CSV non fosse presente, il notebook lo scarica
automaticamente da UCI tramite `ucimlrepo` (id 602). Tutti gli esperimenti usano un
*random seed* fisso (42) per la piena riproducibilità, con la **stessa** suddivisione
train/validation/test (60/20/20) per algoritmi shallow e deep learning.

## Riepilogo dei risultati (test set)

| Modello | Accuracy | ROC AUC (OvR macro) |
|---|---|---|
| **MLP (Deep Learning)** | **0.9240** | **0.9950** |
| SVM (RBF) | 0.9214 | 0.9946 |
| Logistic Regression | 0.9199 | 0.9943 |
| Random Forest | 0.9166 | 0.9929 |
| Naive Bayes | 0.8957 | 0.9912 |
| K-Nearest Neighbors | 0.9141 | 0.9858 |

Il modello migliore è la **rete neurale MLP** (ROC AUC 0.995), con SVM e Logistic Regression
su prestazioni pressoché equivalenti — coerentemente con l'ottima separabilità evidenziata
dalla PCA. La confusione residua riguarda soprattutto le varietà geometricamente simili
(*SIRA* e *DERMASON*), mentre *BOMBAY* è riconosciuta quasi perfettamente.

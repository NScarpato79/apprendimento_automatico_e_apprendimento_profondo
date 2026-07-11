# Elaborato — Apprendimento Automatico e Apprendimento Profondo

**Studente:** Daniele Di Vito — **Matricola:** 0322500152

Esperimento di classificazione multi-classe sul dataset UCI n. 869
[*Shell Commands Used by Participants of Hands-on Cybersecurity Training*](https://archive.ics.uci.edu/dataset/869/shell+commands+used+by+participants+of+hands-on+cybersecurity+training):
riconoscimento dello scenario di training (7 classi) a partire dai comandi shell osservati.

## Contenuto

| File | Descrizione |
|---|---|
| `daniele_di_vito_0322500152.ipynb` | Notebook con l'intero esperimento (PCA, 3 classificatori, metriche, confusion matrix, curve ROC, SHAP) |
| `Relazione_Daniele_Di_Vito_0322500152.pdf` | Relazione di progetto con scelte progettuali, risultati e codice in appendice |
| `shell_commands.csv` | Dataset consolidato (generato dal notebook a partire da `data.zip`) |
| `data.zip` | Archivio originale del dataset (Zenodo, DOI 10.5281/zenodo.8136017) |
| `img/` | Immagini prodotte dal notebook |
| `risultati_metriche.csv` | Tabella delle metriche sul test set |

## Esecuzione

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace daniele_di_vito_0322500152.ipynb
```

Il notebook carica `shell_commands.csv` se presente; in caso contrario scarica e
converte automaticamente `data.zip` da Zenodo. Tutti gli esperimenti usano
`random_state=42` e sono riproducibili.

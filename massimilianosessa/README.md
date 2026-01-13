ELABORATO DI: MASSIMILIANO SESSA
CORSO DI: APPRENDIMENTO AUTOMATICO E APPRENDIMENTO PROFONDO
DOCENTE: NOEMI SCARPATO
ANNO ACCADEMICO: 2025/2026

-------------------------------
Parkinson's Disease Classification & Analysis
Questo repository contiene un notebook Python per l'analisi esplorativa e la classificazione di pazienti affetti dal morbo di Parkinson basandosi su misurazioni biomediche della voce addestrato con un dataset sul morbo di Parkinson disponibile qui --> https://archive.ics.uci.edu/dataset/174/parkinsons

Il progetto include tecniche di riduzione della dimensionalità (PCA), visualizzazione dei dati e un confronto tra diversi algoritmi di Machine Learning.

Istruzioni di Avvio (Input Utente)
----------------------------------
Avviare tutte le celle dall'alto verso il basso facendo attenzione ad eseguire l'installazione del repo con il comando %pip install ucimlrepo che potrebbe restituire qualche warning su Pandas non bloccante.

Eseguire tutte le celle (Import, function e main)
*************************************************************************************************************
Inserire l'input 
-   0 BASELINE per utilizzare un dizionario con i parametri dei 4 modelli con configurazioni di default;
-   1 TEST IPERPARAMETRI per provare dei setup degli iperparametri modificando i valori
*************************************************************************************************************

Nota: Se viene inserito un valore diverso o non numerico, il sistema caricherà automaticamente la modalità Baseline (0).

Caratteristiche Principali

--------------
- Caricamento Dati
--------------
lo script cerca un dataset locale chiamato nel path "dataset/parkinsons.data"
Lo script è progettato per funzionare anche in assenza della cartella o del csv locale, infatti in tale casistica, scaricherà il file dal repo come Fallback.
Per gestire la differenza tra i 2 file (differenze tra i nomi delle features dei campi) normalizza i nomi delle colonne per garantire la compatibilità senza alterare l'eventuale dataset locale.

--------------
- Analisi Esplorativa (EDA)
--------------
PCA (Principal Component Analysis): Riduzione a 2 dimensioni per visualizzare la separabilità delle classi (Sano vs Parkinson) e analisi dei Loadings (peso delle feature sui componenti).
Per approfondire il funzionamento della PCA estraggo le PC prima e dopo la trasposizione e successivamente evidenzio quelle più rilevanti mostrandole in output e creando una heatmap con la matrice di correlazione.

--------------
- Modelli di Machine Learning
--------------
Vengono addestrati e comparati i seguenti classificatori:
-   Logistic Regression
-   Support Vector Machine (SVM)
-   Random Forest
-   Deep Learning (MLP Classifier)

--------------
- Valutazione
--------------
L'output include:
-   Curve ROC: Sovrapposte per un confronto diretto dell'AUC.
-   Matrici di Confusione: Generate per ogni modello per valutare Falsi Positivi/Negativi e successivamente inserite in un unico plot per metterle a confronto.
-   Classification Report: Precision, Recall e F1-Score stampati a terminale.

----------------------------
Requisiti e Installazione
----------------------------

Le librerie necessarie sono inserite nella cella di import.
Come unica accortezza va installato il repo di uci (vedi cella 1 e inizio del file readme).
Opzionale presenza del dataset nel percorso "dataset/parkinsons.data"
Il notebook è stato realizzato con una versione di Python > 3.10

Massimiliano Sessa 13/01/2026
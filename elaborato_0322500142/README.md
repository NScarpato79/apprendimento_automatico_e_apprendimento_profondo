# LM-32

## Apprendimento Automatico e Apprendimento Profondo

### Progetto Machine Learning e Deep Learning - Website Phishing

**Matricola:** 0322500142

**Prof.ssa Noemi Scarpato**

## Notebook Colab

Il progetto puo' essere eseguito anche su Google Colab:

<https://colab.research.google.com/drive/1cNaFaGYYKQ9Ens7aoQnffP9d0uW7a1IL?usp=sharing>

## Installazione Locale

Si consiglia l'utilizzo di Miniconda.

1. Installare Miniconda:

   <https://docs.conda.io/en/latest/miniconda.html>

2. Creare l'ambiente Conda:

   ```bash
   conda env create -f environment.yml
   conda activate ai
   ```

3. In alternativa, creare manualmente l'ambiente:

   ```bash
   conda create -n ai python=3.11 -y
   conda activate ai
   python -m pip install -r requirements.txt
   ```

4. Registrare il kernel Jupyter:

   ```bash
   python -m ipykernel install --user --name ai --display-name "Python (ai)"
   ```

5. Avviare Jupyter Notebook:

   ```bash
   jupyter notebook elaborato.ipynb
   ```

6. Selezionare il kernel **Python (ai)** ed eseguire tutte le celle del notebook.

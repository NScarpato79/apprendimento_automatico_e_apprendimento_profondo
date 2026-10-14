import pandas as pd

# 1. Definisco i nomi ufficiali delle colonne dello Shuttle
nomi_colonne = [
    'Sensore_1', 'Sensore_2', 'Sensore_3', 'Sensore_4',
    'Sensore_5', 'Sensore_6', 'Sensore_7', 'Sensore_8',
    'Sensore_9', 'Classe'
]

print("--- CARICAMENTO LOCALE DEL FILE DI TEST  ---")

# Leggiamo solo il file .tst che è in formato testo
dataset_shuttle = pd.read_csv('shuttle.tst', sep=r'\s+', names=nomi_colonne)

print("File letto con successo.")

# 2. Salvo il dataset in un formato CSV standard
dataset_shuttle.to_csv('shuttle_completo.csv', index=False)
print("File salvato nella cartella del progetto come: 'shuttle_completo.csv'")

# 3. Ispezione dei dati caricati
print("\n--- INFO SUL DATASET CARICATO ---")
print(f"Numero totale di righe (campioni): {dataset_shuttle.shape[0]}")
print(f"Numero totale di colonne (feature + classe): {dataset_shuttle.shape[1]}")

print("\nPrime 5 righe delle letture dei sensori dello Shuttle:")
print(dataset_shuttle.head())

print("\nDistribuzione delle classi (quante anomalie ci sono in queste 14.500 righe):")
print(dataset_shuttle['Classe'].value_counts().sort_index())

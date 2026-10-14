import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("--- TASK 5.a: IMPLEMENTAZIONE APPROACH DI DEEP LEARNING (MLP) ---")

# 1. Carico i dati
df = pd.read_csv('shuttle_completo.csv')
X = df.drop(columns=['Classe'])
y = df['Classe']

# Estraggo il Training (70%), Validation (15%) e Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Standardizzazione per evitare che il gradiente esploda o svanisca
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 2. Configurazione dell'architettura della Rete Neurale (Multi-Layer Perceptron)
print("\nInizializzazione della Rete Neurale Profonda...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Due strati nascosti: 64 neuroni e poi 32 neuroni
    activation='relu',            # Funzione di attivazione ReLU per introdurre non-linearità
    solver='adam',                # Ottimizzatore Adam (il top studiato al Capitolo 33)
    max_iter=500,                 # Numero massimo di epoche di addestramento
    random_state=42,              # Blocco del seed casuale per la riproducibilità
    verbose=True                  # Stampa la perdita (loss) a ogni epoca per monitorare il progresso
)

# 3. Addestramento della Rete Neurale
print("Inizio addestramento (Backpropagation in corso)...")
mlp.fit(X_train_scaled, y_train)

# 4. Valutazione sul Test Set per calcolare le metriche di confronto
y_pred_mlp = mlp.predict(X_test_scaled)

acc_mlp = accuracy_score(y_test, y_pred_mlp)
prec_mlp = precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
rec_mlp = recall_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0)

# 5. Stampa del report finale da inserire nelle conclusioni della tua relazione
print("\n================ METRICHE FINALI - DEEP LEARNING (MLP) ================")
print(f"Accuracy  DL: {acc_mlp*100:.2f}%")
print(f"Precision DL: {prec_mlp*100:.2f}%")
print(f"Recall    DL: {rec_mlp*100:.2f}%")
print(f"F-Measure DL: {f1_mlp*100:.2f}%")
print("=======================================================================")


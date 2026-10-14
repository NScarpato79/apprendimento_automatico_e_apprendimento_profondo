import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("--- TASK 2 & 3: ADDESTRAMENTO MODELLI E CALCOLO METRICHE ---")

# 1. Preparo i dati  come nel Task 1
df = pd.read_csv('shuttle_completo.csv')
X = df.drop(columns=['Classe'])
y = df['Classe']

# Rieseguo lo split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Standardizzo i dati (per Logistic Regression e KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Inizializzo i 3 algoritmi scelti
modelli = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Dizionario per salvare i risultati da stampare in una tabella per la relazione
risultati = []

# 3. Loop per addestrare e valutare ogni modello
for nome, modello in modelli.items():
    print(f"\nAddestramento di {nome} in corso...")
    modello.fit(X_train_scaled, y_train)

    # Predizioni delle classi
    y_pred = modello.predict(X_test_scaled)
    # Predizioni delle probabilità (necessarie per la ROC AUC)
    y_proba = modello.predict_proba(X_test_scaled)

    # Calcolo delle metriche (uso average='macro' o 'weighted' perché è un problema multiclasse)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Per la ROC AUC multiclasse uso l'approccio One-Vs-Rest (ovr)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = np.nan  # Protezione in caso qualche classe mancasse nel subset di test

    risultati.append({
        "Modello": nome,
        "Accuracy": f"{acc * 100:.2f}%",
        "Precision": f"{prec * 100:.2f}%",
        "Recall": f"{rec * 100:.2f}%",
        "F-Measure": f"{f1 * 100:.2f}%",
        "ROC AUC": f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    })

# 4. Mostro i risultati finali in una tabella
df_risultati = pd.DataFrame(risultati)
print("\n================ TABELLA COMPARATIVA DELLE METRICHE ================")
print(df_risultati.to_string(index=False))
print("====================================================================")

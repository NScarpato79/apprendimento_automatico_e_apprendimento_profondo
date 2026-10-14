import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("--- TASK 4: GENERAZIONE MATRICE DI CONFUSIONE E CURVA ROC ---")

# 1. Preparo i dati
df = pd.read_csv('shuttle_completo.csv')
X = df.drop(columns=['Classe'])
y = df['Classe']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Alleno il modello: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)

# --- PARTE A: MATRICE DI CONFUSIONE ---
print("Generazione della Matrice di Confusione...")
cm = confusion_matrix(y_test, y_pred)
# classi presenti nel test set per evitare sfasamenti nei grafici
classi_presenti = np.unique(y_test)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi_presenti)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Matrice di Confusione - Random Forest (Shuttle)', fontsize=14)

# Salviamo l'immagine per la relazione
plt.savefig('shuttle_confusion_matrix.png', dpi=300)
print("Matrice di confusione salvata come 'shuttle_confusion_matrix.png'")

# --- PARTE B: CURVA ROC MULTICLASSE (Risoluzione del problema N/A) ---
print("Generazione della Curva ROC...")
y_test_bin = label_binarize(y_test, classes=classi_presenti)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(9, 7))

# Calcolo e disegno la curva ROC per ogni classe disponibile
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Classe {classi_presenti[i]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Curve ROC per Classe (Approccio One-Vs-Rest)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)

# Salvo la seconda immagine per la relazione
plt.savefig('shuttle_roc_curve.png', dpi=300)
print("Curva ROC salvata come 'shuttle_roc_curve.png'")

# Mostro i grafici a schermo
plt.show()

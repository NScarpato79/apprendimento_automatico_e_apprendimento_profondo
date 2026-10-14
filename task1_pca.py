import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("--- TASK 1: DIVISIONE DATI, STANDARDIZZAZIONE E PCA ---")

# 1. Carico il dataset
df = pd.read_csv('shuttle_completo.csv')

# Separo le feature (X, i 9 sensori) dalla variabile target (y, la classe di anomalia)
X = df.drop(columns=['Classe'])
y = df['Classe']

# 2. Suddivisione manuale
# Primo split: separo il 70% di Training dal restante 30%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

# Secondo split: divido il 30% temporaneo a metà (15% Validation e 15% Test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Dataset originale splitto manualmente in:")
print(f" - Training Set:   {X_train.shape} campioni (70%)")
print(f" - Validation Set: {X_val.shape} campioni (15%)")
print(f" - Test Set:       {X_test.shape} campioni (15%)")

# 3. Standardizzazione (prima della PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Applicazione della PCA (riduco da 9 dimensioni a 2 per visualizzarle)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

varianza_spiegata = np.sum(pca.explained_variance_ratio_) * 100
print(f"\n📈 Le prime 2 Componenti Principali spiegano il {varianza_spiegata:.2f}% della varianza totale dei sensori!")

# 5. Creazione del Grafico 2D della PCA
plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_train_pca[:, 0],
    y=X_train_pca[:, 1],
    hue=y_train,
    palette='viridis',
    alpha=0.7,
    legend='full'
)

plt.title('Analisi PCA dello Space Shuttle Dataset (2 Componenti)', fontsize=14)
plt.xlabel('Componente Principale 1 (PC1)', fontsize=12)
plt.ylabel('Componente Principale 2 (PC2)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Classe Anomalia')

# Salvo il grafico come immagine sul PC
plt.savefig('shuttle_pca_plot.png', dpi=300)
print("Grafico salvato nella cartella del progetto come 'shuttle_pca_plot.png'.")

# Mostro il grafico a schermo
plt.show()

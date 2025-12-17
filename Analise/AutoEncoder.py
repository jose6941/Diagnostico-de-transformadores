import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    log_loss, cohen_kappa_score, roc_curve, auc
)
import seaborn as sns
from itertools import cycle

# Carregar o dataset
input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
names = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'defeito']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

df = pd.read_csv(input_file, names=names)

X = df[features].values
y = df[target].values

# Encoding dos rótulos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# "Autoencoder" com PCA (reduzindo para 10 componentes)
latent_dim = min(X_train.shape[1], 5)  # Usa 5 ou menos, dependendo da base
pca = PCA(n_components=latent_dim)
X_train_latent = pca.fit_transform(X_train)
X_test_latent = pca.transform(X_test)

# Gráfico de variância explicada
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Variância Explicada Acumulada (PCA)')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada')
plt.grid(True)
plt.show()

# Classificador sobre espaço latente
clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=1000, random_state=42)
clf.fit(X_train_latent, y_train)

# Predições
y_pred = clf.predict(X_test_latent)
y_pred_proba = clf.predict_proba(X_test_latent)

# Avaliações
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
kappa = cohen_kappa_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"\nAcurácia: {acc*100:.2f}%")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Cohen’s Kappa: {kappa:.4f}")
print(f"Log Loss: {loss:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusão - PCA + MLPClassifier")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Curvas ROC / AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title("Curvas ROC - PCA + MLPClassifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

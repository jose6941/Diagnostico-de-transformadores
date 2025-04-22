import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

# Carregar dados
input_file = 'Dataset_sintetico/dados_sinteticos_normalized.csv'
names = ['H2','CH4','C2H2','C2H4','C2H6','defeito']
features = ['H2','CH4','C2H2','C2H4','C2H6'] 
target = 'defeito'

df = pd.read_csv(input_file, names=names)

X = df[features].values
y = df[target].values

# Codificar rótulos se necessário
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Treinar modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Prever
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
kappa = cohen_kappa_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"\nAcurácia: {accuracy * 100:.2f}%")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Cohen’s Kappa: {kappa:.4f}")
print(f"Log Loss: {loss:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in class_names]))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusão - Naive Bayes")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ROC/AUC Multiclasse
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC - Naive Bayes Multiclasse")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

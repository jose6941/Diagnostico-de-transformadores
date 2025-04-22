import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    log_loss, cohen_kappa_score, roc_curve, auc
)
from itertools import cycle

input_file = 'Datasets/GasesDissolvidos_Normalized.csv'
names = ['H2','CH4','C2H2','C2H4','C2H6','defeito']
features = ['H2','CH4','C2H2','C2H4','C2H6'] 
target = 'defeito'

df = pd.read_csv(input_file, names=names) 
X = df[features].values
y = df[target].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

class AttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        attention_weights = torch.softmax(self.attention(x), dim=1)
        x = x * attention_weights
        x = self.fc2(x)
        return x

input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(np.unique(y))

model = AttentionClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    train_losses.append(epoch_train_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    model.eval()
    epoch_test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == y_batch).sum().item()
            total_test += y_batch.size(0)
    test_losses.append(epoch_test_loss / len(test_loader))
    test_accuracies.append(correct_test / total_test)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, "
          f"Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

# Gráficos de perda e acurácia
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Perda de Treinamento")
plt.plot(test_losses, label="Perda de Teste")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.legend()
plt.title("Perda durante o Treinamento")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Acurácia de Treinamento")
plt.plot(test_accuracies, label="Acurácia de Teste")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.title("Acurácia durante o Treinamento")
plt.show()

model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).detach().cpu().numpy()
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_labels, target_names=[str(cls) for cls in label_encoder.classes_]))

print(f"Log Loss: {log_loss(y_test, y_pred_probs):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred_labels):.4f}")

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

y_test_bin = label_binarize(y_test, classes=np.unique(y))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['darkorange', 'green', 'blue', 'red', 'purple', 'cyan'])
plt.figure(figsize=(10, 7))
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Classe {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC para Classificação Multiclasse')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

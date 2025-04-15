import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = 'Datasets/GasesDissolvidos_Normalized.csv'
names = ['H2','CH4','C2H2','C2H4','C2H6','defeito']
features = ['H2','CH4','C2H2','C2H4','C2H6'] 
target = 'defeito'

df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas 

X = df[features].values
y = df[target].values

# Transformação do alvo em categorias
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Conversão para tensores PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader para facilitar o treinamento em batches
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Definição do Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Definição do Classificador
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Parâmetros do modelo
input_size = X_train.shape[1]
latent_dim = 10  # Dimensão da camada latente
num_classes = len(np.unique(y))

# Inicialização dos modelos
autoencoder = Autoencoder(input_size, latent_dim)
classifier = Classifier(latent_dim, num_classes)

# Funções de perda e otimizadores
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Treinamento do Autoencoder
num_epochs_ae = 50
train_losses_ae = []
for epoch in range(num_epochs_ae):
    autoencoder.train()
    epoch_loss = 0
    for X_batch, _ in train_loader:
        ae_optimizer.zero_grad()
        reconstructed, _ = autoencoder(X_batch)
        loss = reconstruction_criterion(reconstructed, X_batch)
        loss.backward()
        ae_optimizer.step()
        epoch_loss += loss.item()
    train_losses_ae.append(epoch_loss / len(train_loader))
    print(f"Autoencoder Epoch {epoch+1}/{num_epochs_ae}, Loss: {train_losses_ae[-1]:.4f}")

# Plot da perda do Autoencoder
plt.figure(figsize=(10, 5))
plt.plot(train_losses_ae, label="Perda do Autoencoder")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Treinamento do Autoencoder")
plt.legend()
plt.show()

# Treinamento do Classificador
num_epochs_clf = 20
train_losses_clf, test_losses_clf = [], []
for epoch in range(num_epochs_clf):
    autoencoder.eval()
    classifier.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        clf_optimizer.zero_grad()
        _, latent = autoencoder(X_batch)
        outputs = classifier(latent)
        loss = classification_criterion(outputs, y_batch)
        loss.backward()
        clf_optimizer.step()
        epoch_loss += loss.item()
    train_losses_clf.append(epoch_loss / len(train_loader))

    # Avaliação no conjunto de teste
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _, latent = autoencoder(X_batch)
            outputs = classifier(latent)
            loss = classification_criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
    test_losses_clf.append(test_loss / len(test_loader))
    accuracy = correct / len(test_data)
    print(f"Classifier Epoch {epoch+1}/{num_epochs_clf}, Train Loss: {train_losses_clf[-1]:.4f}, Test Loss: {test_losses_clf[-1]:.4f}, Accuracy: {accuracy:.4f}")

# Plot das perdas do classificador
plt.figure(figsize=(10, 5))
plt.plot(train_losses_clf, label="Treino")
plt.plot(test_losses_clf, label="Teste")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Treinamento do Classificador")
plt.legend()
plt.show()
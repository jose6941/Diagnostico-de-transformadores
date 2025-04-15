import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = 'Dataset_sintetico/dados_sinteticos_normalized.csv'
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

# Expansão para formato compatível com RNN (batch_size, seq_length, input_size)
X_scaled = np.expand_dims(X_scaled, axis=1)

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

# Definição do modelo RNN
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)  # Saída final do LSTM
        out = self.fc(hidden[-1])     # Passa pela camada totalmente conectada
        return out

# Hiperparâmetros
input_size = X_train.shape[2]
hidden_size = 64
output_size = len(np.unique(y))
num_epochs = 20
learning_rate = 0.001

# Inicialização do modelo, função de perda e otimizador
model = RNNClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento
train_losses, test_losses = [], []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    
    # Avaliação no conjunto de teste
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
    test_losses.append(test_loss / len(test_loader))
    accuracy = correct / len(test_data)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

# Plot das perdas
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Treino")
plt.plot(test_losses, label="Teste")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Perda ao Longo das Épocas")
plt.legend()
plt.show()
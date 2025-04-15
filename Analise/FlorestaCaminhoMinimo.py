import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path):
    
    data = pd.read_csv(file_path)
    return data

def create_minimum_spanning_forest(X, y):

    # Calcula a matriz de distância
    distance_matrix = squareform(pdist(X, metric='euclidean'))
    
    # Cria a árvore de caminhos mínimos
    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    # Adiciona rótulos aos nós
    labeled_graph = defaultdict(list)
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i, j] > 0:  # Apenas as conexões da MST
                labeled_graph[i].append((j, mst[i, j]))
    
    return labeled_graph

def classify_with_forest(graph, X_train, y_train, X_test, k=1):
    
    # Usando KNN para a classificação (exemplo simples para uso da floresta)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predição
    predictions = knn.predict(X_test)
    return predictions

file_path = "Datasets/GasesDissolvidos_Normalized.csv"  # Substitua pelo caminho do seu arquivo

data = load_data(file_path)

X = data.iloc[:, :-1].values  # Atributos (todas as colunas menos a última)
y = data.iloc[:, -1].values   # Rótulos (última coluna)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
forest = create_minimum_spanning_forest(X_train, y_train)
y_pred = classify_with_forest(forest, X_train, y_train, X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")
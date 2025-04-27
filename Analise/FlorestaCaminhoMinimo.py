import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from itertools import cycle
import seaborn as sns

def load_data(file_path):
    data = pd.read_csv(file_path, names=['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'defeito'])
    return data

def create_minimum_spanning_forest(X):
    distance_matrix = squareform(pdist(X, metric='euclidean'))
    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    labeled_graph = defaultdict(list)
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i, j] > 0:
                labeled_graph[i].append((j, mst[i, j]))
    return labeled_graph

def classify_with_forest(X_train, y_train, X_test, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    probs = knn.predict_proba(X_test)
    return predictions, probs

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.show()

def plot_roc(y_test_bin, y_pred_proba, class_names, title='Curvas ROC'):
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def main():
    file_path = "Datasets/GasesDissolvidos_Normalized.csv"
    data = load_data(file_path)

    features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    target = 'defeito'

    X = data[features].values
    y = data[target].values

    # Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criação da floresta mínima (não usada diretamente, mas para referência)
    forest = create_minimum_spanning_forest(X_train)

    # Classificação via KNN
    y_pred, y_pred_proba = classify_with_forest(X_train, y_train, X_test, k=3)

    # Avaliação
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
    plot_confusion_matrix(cm, class_names, normalize=False, title="Matriz de Confusão - MST + KNN")
    plot_confusion_matrix(cm, class_names, normalize=True, title="Matriz de Confusão Normalizada - MST + KNN")

    # Curvas ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    plot_roc(y_test_bin, y_pred_proba, class_names, title="Curvas ROC - MST + KNN")

if __name__ == "__main__":
    main()

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from itertools import cycle

def minkowski_distance(a, b, p=1):    
    dim = len(a)    
    distance = sum(abs(a[d] - b[d])**p for d in range(dim))
    return distance**(1/p)

def knn_predict(X_train, X_test, y_train, k, p):    
    y_hat_test = []
    for test_point in X_test:
        distances = [(minkowski_distance(test_point, train_point, p=p), label)
                     for train_point, label in zip(X_train, y_train)]
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
        prediction = Counter(k_nearest_labels).most_common(1)[0][0]
        y_hat_test.append(prediction)
    return y_hat_test

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc(y_test_bin, y_proba, class_names, title):
    n_classes = y_test_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito']
    features = ['H2','CH4','C2H2','C2H4','C2H6']
    target = 'defeito'

    data = pd.read_csv(input_file, names=names)

    X = data[features].values
    y = data[target].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    print("Total samples: {}".format(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    print("\n----- KNN FROM SCRATCH -----")
    y_hat_test = knn_predict(X_train, X_test, y_train, k=5, p=2)

    acc = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy: {:.2f}%".format(acc))
    print("F1 Score: {:.4f}".format(f1))

    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, class_names, False, "Matriz de Confusão - KNN Scratch")
    plot_confusion_matrix(cm, class_names, True, "Matriz de Confusão Normalizada - KNN Scratch")

    # -------------------------------
    print("\n----- KNN SKLEARN -----")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)
    y_hat_test_proba = knn.predict_proba(X_test)

    acc = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    kappa = cohen_kappa_score(y_test, y_hat_test)
    loss = log_loss(y_test, y_hat_test_proba)

    print("Accuracy: {:.2f}%".format(acc))
    print("F1 Score: {:.4f}".format(f1))
    print("Cohen's Kappa: {:.4f}".format(kappa))
    print("Log Loss: {:.4f}".format(loss))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_hat_test, target_names=[str(cls) for cls in label_encoder.classes_]))

    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, class_names, False, "Matriz de Confusão - KNN Sklearn")
    plot_confusion_matrix(cm, class_names, True, "Matriz de Confusão Normalizada - KNN Sklearn")

    # Curvas ROC/AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    plot_roc(y_test_bin, y_hat_test_proba, class_names, "Curvas ROC - KNN Sklearn")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.neural_network import MLPClassifier
from itertools import cycle

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def main():
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6','defeito']
    features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    target = 'defeito'

    df = pd.read_csv(input_file, names=names)

    X = df[features].values
    y = df[target].values
    print("Total samples: {}".format(X.shape[0]))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tamanho_camadas = [64, 32]
    mlp = MLPClassifier(hidden_layer_sizes=tamanho_camadas, max_iter=1000,
                        random_state=1, activation='relu', alpha=0.01)
    mlp.fit(X_train, y_train)

    # Validação Cruzada
    cv_scores = cross_val_score(mlp, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score: {:.2f}%".format(cv_scores.mean() * 100))
    print("Standard deviation of cross-validation scores: {:.2f}".format(cv_scores.std()))

    y_hat_test = mlp.predict(X_test)
    y_hat_test_proba = mlp.predict_proba(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    kappa = cohen_kappa_score(y_test, y_hat_test)
    loss = log_loss(y_test, y_hat_test_proba)

    print("\nAccuracy: {:.2f}%".format(accuracy))
    print("F1 Score (macro): {:.4f}".format(f1))
    print("Cohen’s Kappa: {:.4f}".format(kappa))
    print("Log Loss: {:.4f}".format(loss))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_hat_test, target_names=[str(cls) for cls in label_encoder.classes_]))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, class_names, normalize=False, title="Matriz de Confusão - MLP")
    plot_confusion_matrix(cm, class_names, normalize=True, title="Matriz de Confusão Normalizada - MLP")

    # ROC/AUC Multiclasse
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_hat_test_proba[:, i])
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
    plt.title("Curvas ROC - MLPClassifier")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

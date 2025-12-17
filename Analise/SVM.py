import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.svm import SVC
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

def load_dataset(file_path):        
    names = ['H2','CH4','C2H2','C2H4','C2H6', 'defeito'] 
    features = ['H2','CH4','C2H2','C2H4','C2H6'] 
    target = 'defeito'
    df = pd.read_csv(file_path, names=names)
    print(df.head())
    return df, features, target

def main():
    # Load dataset
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    df, features, target = load_dataset(input_file)

    # Separate X and y data
    X = df[features]
    y = df[target]
    print("Total samples: {}".format(X.shape[0]))

    # Encode labels if necessary
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=1)
    
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM classifier
    svm = SVC(kernel='poly', probability=True)
    svm.fit(X_train, y_train)

    print("Qtd Support vectors:", svm.n_support_)

    y_hat_test = svm.predict(X_test)
    y_hat_test_proba = svm.predict_proba(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    kappa = cohen_kappa_score(y_test, y_hat_test)
    loss = log_loss(y_test, y_hat_test_proba)

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Cohen’s Kappa: {kappa:.4f}")
    print(f"Log Loss: {loss:.4f}")

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_hat_test, target_names=[str(cls) for cls in label_encoder.classes_]))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, classes=class_names, normalize=False, title="Matriz de Confusão - SVM")
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title="Matriz de Confusão Normalizada - SVM")

    # Curvas ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
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
    plt.title("Curvas ROC - Classificação Multiclasse - SVM")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

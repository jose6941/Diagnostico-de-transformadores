import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
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
    input_file = 'Datasets/GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito']
    features = ['H2','CH4','C2H2','C2H4','C2H6'] 
    target = 'defeito'

    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas    

    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    X = df[features].values
    y = df[target].values
    print("Total samples: {}".format(X.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    tamanho_camadas = [64,32]
    mlp = MLPClassifier(hidden_layer_sizes=tamanho_camadas, max_iter=1000, random_state=1, activation='relu', alpha=0.01)
    mlp.fit(X_train, y_train)

    cv_scores = cross_val_score(mlp, X, y, cv=5)  # 5-fold cross-validation
    print("Cross-validation scores: ", cv_scores)
    print("Mean cross-validation score: {:.2f}%".format(cv_scores.mean() * 100))
    print("Standard deviation of cross-validation scores: {:.2f}".format(cv_scores.std()))

    y_hat_test = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy MLPClassifier: {:.2f}%".format(accuracy))
    print("F1 Score MLPClassifier: {:.2f}".format(f1))

    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, [1,2,3],False, "Confusion Matrix - MLPClassifier")      
    plot_confusion_matrix(cm, [1,2,3],True, "Confusion Matrix - MLPClassifier normalized")  
    plt.show()

if __name__ == "__main__":
    main()
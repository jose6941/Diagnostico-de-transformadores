import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Conjunto de dados

home_data = pd.read_csv('Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv', 
                        names = ['H2','CH4','C2H2','C2H4','C2H6','defeito'])
home_data.head()

#Visualização dos dados
sns.scatterplot(data = home_data, x = 'H2', y = 'CH4', hue = 'defeito')

#Normalização dos dados
X_train, X_test, y_train, y_test = train_test_split(home_data[['H2', 'CH4']],
                                                    home_data[['defeito']], 
                                                    test_size=0.33, random_state=0)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

K = range(2, 8)
fits = []
score = []

#Escolhendo o número de clusters
for k in K:
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
    fits.append(model)
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

sns.scatterplot(data = X_train, x = 'H2', y = 'CH4', hue = fits[2].labels_)
#sns.lineplot(x = K, y = score)
plt.show()


#Ajuste e avalição do modelo
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
sns.scatterplot(data = X_train, x = 'H2', y = 'CH4', hue = kmeans.labels_)
sns.boxplot(x = kmeans.labels_, y = y_train['defeito'])
silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

plt.show()
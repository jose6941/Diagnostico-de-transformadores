import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito'] 
    features = ['H2','CH4','C2H2','C2H4','C2H6'] 
    target = 'defeito'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalizedDf,"Dataframe Normalized")

    # Cria PCA
    pca = PCA(n_components=3)    
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(data = principalComponents[:,0:3], 
                               columns = ['principal component 1', 
                                          'principal component 2',
                                          'principal component 3'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)    
    ShowInformationDataFrame(finalDf,"Dataframe PCA")
    VisualizePcaProjection(finalDf, target)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")
    
#Exibe PCA
def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8)) 
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('Componente 1', fontsize = 15)
    ax.set_ylabel('Componente 2', fontsize = 15)
    ax.set_zlabel('Componente 3', fontsize = 15)
    ax.set_title('PCA com 3 componentes', fontsize = 20)
    targets = [1, 2, 3, 4, 5, 6, 7]
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   c = color, s = 50)
    
    ax.legend(targets)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()
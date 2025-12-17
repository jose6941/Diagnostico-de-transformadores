import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def main():
    # Faz a leitura do arquivo
    names = ['defeito','H2','CH4','C2H2','C2H4','C2H6'] 
    target = 'defeito'
    features = ['H2','CH4','C2H2','C2H4','C2H6']
    output_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Clear.csv'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    x = df.loc[:, features].values
    y = df.loc[:,[target]].values

    if input('Normalization: [1]Z-Score, [2]Min-Max > ') == 1:
        # Z-score normalization
        x_zcore = StandardScaler().fit_transform(x)
        normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
        normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
        ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")
        # Salva arquivo com normalização Z-Score
        normalized1Df.to_csv(output_file, header=False, index=False)  
    else:
        # Min-Max normalization
        x_minmax = MinMaxScaler().fit_transform(x)
        normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
        normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
        ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")
        # Salva arquivo com normalização Min-Max
        normalized2Df.to_csv(output_file, header=False, index=False)  

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

if __name__ == "__main__":
    main()
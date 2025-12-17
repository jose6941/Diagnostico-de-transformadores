import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito'] 
    features = ['H2','CH4','C2H2','C2H4','C2H6'] 
    target = 'defeito'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes    

    N = int(input('Digite a quantidade de Bins: '))
    freqtable = pd.cut(df['H2'], bins=N) 
    freq = freqtable.value_counts().sort_index()
    print(freq)

    #Exibe a tabela
    plt.bar(freq.index.astype(str), freq.values)
    plt.xlabel('Categorias')
    plt.ylabel('Frequencias')
    plt.title('Distribuição de Frequência')

    plt.show()

if __name__ == "__main__":
    main()
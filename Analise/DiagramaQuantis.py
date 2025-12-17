import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito'] 
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      

    atr = df['H2']
    print(atr.describe())
    fig = plt.boxplot(atr)
    plt.show()

if __name__ == "__main__":
    main()
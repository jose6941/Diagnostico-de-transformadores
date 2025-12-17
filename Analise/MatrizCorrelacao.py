import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    input_file = 'Diagnostico-de-transformadores\Datasets\GasesDissolvidos_Normalized.csv'
    names = ['H2','CH4','C2H2','C2H4','C2H6','defeito'] 
    df = pd.read_csv(input_file,    
                     names = names)                       

    correlacao = df.corr()
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Matriz de Correlação")
    plt.show()

if __name__ == "__main__":
    main()
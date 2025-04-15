import pandas as pd
import numpy as np

df = pd.read_csv("Dados.csv")

colunas = ["H2","CH4","C2H2","C2H4","C2H6"]

medias = []
padrao = []
minimo = []
maximo = []
valores = {}
defeitos_sinteticos = np.random.choice(df["defeito"].unique(), 1000, p=df["defeito"].value_counts(normalize=True))

def gerar_dados(coluna):
   medias = df[coluna].mean()
   padrao = df[coluna].std()*0.8
   minimo = df[coluna].min()
   maximo = df[coluna].max()*0.8

   dados = np.random.normal(medias, padrao, 1000)
   valor = np.clip(dados, minimo, maximo)
   return valor

valores = {"defeito": defeitos_sinteticos}

for coluna in colunas:
   valores[coluna] = gerar_dados(coluna)

tabela = pd.DataFrame(valores)
tabela.to_csv("dados_sinteticos.csv", index=False)


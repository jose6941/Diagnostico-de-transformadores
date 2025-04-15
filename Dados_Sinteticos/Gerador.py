import numpy as np
import pandas as pd
import time
import datetime
import os

file_path = "Dados_Sinteticos/Dados.csv"  
df = pd.read_csv(file_path)  

colunas_gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6"]

num_samples = len(df)  
np.random.seed(int(time.time_ns()) % 2**32)  

defeitos_possiveis = df["defeito"].unique()
defeitos_probabilidades = df["defeito"].value_counts(normalize=True)
defeitos_sinteticos = np.random.choice(defeitos_possiveis, num_samples, p=defeitos_probabilidades.values)

def gerar_dado_sintetico_proporcional(coluna, defeito, usados):
    df_defeito = df[df["defeito"] == defeito]
    if df_defeito.empty:
        df_defeito = df
    valores_unicos = df_defeito[coluna].drop_duplicates().values
    np.random.shuffle(valores_unicos)
    for valor in valores_unicos:
        if valor not in usados:
            fator_ruido = np.random.normal(0, 0.05)
            fator_ruido_extra = np.random.uniform(-0.03, 0.03)
            novo_valor = valor * (1 + fator_ruido + fator_ruido_extra)
            usados.add(valor)
            return novo_valor
    return np.random.choice(valores_unicos)  

sintetico_dict = {"defeito": []}
usados_por_coluna = {col: set(df[col].values) for col in colunas_gases}  
for coluna in colunas_gases:
    sintetico_dict[coluna] = []

for defeito in defeitos_sinteticos:
    sintetico_dict["defeito"].append(defeito)
    for coluna in colunas_gases:
        valor = gerar_dado_sintetico_proporcional(coluna, defeito, usados_por_coluna[coluna])
        sintetico_dict[coluna].append(valor)

df_sintetico = pd.DataFrame(sintetico_dict)
df_combinado = pd.concat([df, df_sintetico], ignore_index=True)
df_combinado = df_combinado.sample(frac=1).reset_index(drop=True)  

file_output = f"Dataset_Sintetico/dados_sinteticos.csv"

output_path = os.path.join(".", file_output)
df_combinado.to_csv(output_path, index=False)

print(f"Base combinada gerada e salva como '{file_output}'")
print(df_combinado.head())

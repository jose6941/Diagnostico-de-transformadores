from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("./modelo_xgboost.joblib")
scaler = joblib.load("./scaler.joblib")

class InputData(BaseModel):
    H2: float
    CH4: float
    C2H2: float
    C2H4: float
    C2H6: float

mapa_defeitos = {
    0: "Sem falha",
    1: "Descarga parcial",
    2: "Superaquecimento Leve do Óleo",
    3: "Superaquecimento Grave do Óleo"
}

descricao_falhas = {
    "Sem falha": "O transformador está operando normalmente, sem indícios de falha.",
    "Descarga parcial": "Fenômeno elétrico causado por imperfeições no isolamento. Pode evoluir para falhas mais graves.",
    "Superaquecimento Leve do Óleo": "O superaquecimento leve está geralmente relacionado a conexões frouxas, resistência de contato elevada ou pontos quentes mal distribuídos. Os gases formados são típicos de degradação térmica inicial, como metano e etano, mas ainda sem presença crítica de etileno.",
    "Superaquecimento Grave do Óleo": "Neste caso, o transformador apresenta áreas com temperaturas superiores a 300 °C, o que causa forte degradação térmica do óleo. Etileno passa a ser o gás predominante, e a integridade do sistema isolante começa a ser comprometida.",
}

medidas_preventivas = {
    "Sem falha": "Continuar monitorando regularmente com análise de gases dissolvidos.",
    "Descarga parcial": "A principal forma de evitar descargas parciais é manter o sistema de isolamento em bom estado. Isso envolve o uso de óleo isolante de alta qualidade e com baixa umidade, além de evitar a presença de bolhas de ar. Deve-se garantir que o transformador esteja sempre hermeticamente selado ou com sistemas de vedação bem conservados. Também é fundamental instalar sensores de descarga parcial em transformadores críticos para detecção precoce.",
    "Superaquecimento Leve do Óleo": "É fundamental manter a integridade mecânica das conexões internas e externas, principalmente nas buchas, barramentos e terminais. A sobrecarga do transformador deve ser evitada com base em estudos de carregamento e monitoramento em tempo real. A ventilação adequada e a limpeza de radiadores e serpentinas de resfriamento são medidas preventivas importantes.",
    "Superaquecimento Grave do Óleo": "Evitar o funcionamento prolongado em condições de sobrecarga é essencial. É necessário que o sistema de resfriamento esteja em pleno funcionamento: radiadores limpos, ventiladores operando corretamente e óleo com boas propriedades térmicas. A instalação de sensores de temperatura e dispositivos de alarme ajuda a identificar elevações anormais de temperatura em tempo real.",
}

@app.post("/diagnostico")
def diagnosticar(data: InputData):
    entrada = np.array([[data.H2, data.CH4, data.C2H2, data.C2H4, data.C2H6]])
    entrada_normalizada = scaler.transform(entrada)
    pred = model.predict(entrada_normalizada)
    codigo = int(pred[0])
    falha = mapa_defeitos.get(codigo, "Desconhecido")
    return {
        "codigo": codigo,
        "falha": falha,
        "descricao": descricao_falhas.get(falha, "Sem descrição"),
        "medidas": medidas_preventivas.get(falha, "Sem medidas específicas")
    }

@app.get("/informacoes_falhas")
def obter_informacoes():
    return {
        "descricao_falhas": descricao_falhas,
        "medidas_preventivas": medidas_preventivas
    }

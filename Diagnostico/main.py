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
    2: "Falha térmica",
    3: "Arco elétrico",
    4: "Degradação da celulose"
}

descricao_falhas = {
    "Sem falha": "O transformador está operando normalmente, sem indícios de falha.",
    "Descarga parcial": "Fenômeno elétrico causado por imperfeições no isolamento. Pode evoluir para falhas mais graves.",
    "Falha térmica": "Superaquecimento interno causado por sobrecarga ou ventilação deficiente, que degrada o isolamento.",
    "Arco elétrico": "Descarga elétrica de alta energia que gera gases combustíveis em alta concentração e pode danificar severamente o equipamento.",
    "Degradação da celulose": "Indica envelhecimento do papel isolante (celulose), geralmente devido ao calor, umidade ou tempo de operação prolongado."
}

medidas_preventivas = {
    "Sem falha": "Continuar monitorando regularmente com análise de gases dissolvidos (DGA).",
    "Descarga parcial": "Realizar ensaio dielétrico, verificar integridade dos isoladores e possíveis contaminações.",
    "Falha térmica": "Avaliar sobrecargas, verificar sistemas de ventilação e refrigeração e realizar manutenção preventiva.",
    "Arco elétrico": "Inspeção imediata, desligamento programado, verificação de conexões internas e substituição de componentes afetados.",
    "Degradação da celulose": "Revisar condições térmicas e de carga, controlar umidade do óleo e considerar reforma ou substituição do transformador."
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

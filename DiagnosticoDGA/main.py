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

@app.post("/diagnostico")
def diagnosticar(data: InputData):
    entrada = np.array([[data.H2, data.CH4, data.C2H2, data.C2H4, data.C2H6]])
    entrada_normalizada = scaler.transform(entrada)
    pred = model.predict(entrada_normalizada)
    codigo = int(pred[0])
    return {"codigo": codigo, "falha": mapa_defeitos.get(codigo, "Desconhecido")}

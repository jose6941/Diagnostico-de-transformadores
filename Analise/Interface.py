import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Diagnóstico de Transformadores",
    layout="centered"
)

model = joblib.load("joblib/modelo_xgboost.joblib")
scaler = joblib.load("joblib/scaler.joblib")
label_encoder = joblib.load("joblib/label_encoder.joblib")

st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #004466;
        }
        .subtitle {
            text-align: center;
            font-weight: bold;
        }
        .info {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: gray;
            margin-top: 20px;
        }
        .stButton>button {
            display: flex;
            justify-content: center;
            background-color: #004466;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Diagnóstico de Falhas em Transformadores</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Baseado em Análise DGA com XGBoost</div>", unsafe_allow_html=True)

st.markdown("<div class='info'>Informe as concentrações dos gases em ppm</div>",unsafe_allow_html=True)

gases = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
valores = []

col1, col2 = st.columns(2)
for i, gas in enumerate(gases):
    with (col1 if i < 3 else col2):
        val = st.number_input(f"{gas}", min_value=0.0, step=0.1)
        valores.append(val)

mapa_defeitos = {
    0: "Sem falha",
    1: "Descarga parcial",
    2: "Falha térmica",
    3: "Arco elétrico",
    4: "Degradação da celulose"
}

explicacoes = {
        "Arco": "Descarga de alta energia no óleo isolante.",
        "Falaha térmica": "Sobreaquecimento detectado no enrolamento ou no isolamento.",
        "Descarga parcial": "Fenômeno elétrico que pode indicar degradação do isolamento.",
        "Celulose": "Degradação térmica dos componentes sólidos (papel isolante).",
        "Sem falha": "Transformador em condição de operação normal."
}

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Diagnosticar"):
        entrada = np.array(valores).reshape(1, -1)
        entrada_normalizada = scaler.transform(entrada)
        pred = model.predict(entrada_normalizada)

        classe_id = int(pred[0])
        diagnostico = mapa_defeitos.get(classe_id, "Desconhecido")

        st.success(f"Tipo de falha diagnosticada: **{diagnostico}**")
        st.info(explicacoes.get(diagnostico, "Sem descrição disponível."))



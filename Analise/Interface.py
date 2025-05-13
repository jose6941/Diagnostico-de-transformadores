import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diagnóstico de Transformadores", layout="centered")

model = joblib.load("joblib/modelo_xgboost.joblib")
scaler = joblib.load("joblib/scaler.joblib")

st.markdown("""
    <style>
        .form-card {
            background-color: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            max-width: 650px;
            margin: auto;
        }

        .form-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 20px;
            color: white;
            margin: 10px;
            text-align: center;
        }

        .footer {
            font-size: 13px;
            text-align: center;
            color: gray;
            margin-top: 50px;
        }

        .stButton > button {
            background-color: #004466;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            font-size: 16px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Diagnóstico de Transformadores</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Análise de Gases Dissolvidos no Óleo (DGA) com XGBoost</div>", unsafe_allow_html=True)

mapa_defeitos = {
    0: "Sem falha",
    1: "Descarga parcial",
    2: "Falha térmica",
    3: "Arco elétrico",
    4: "Degradação da celulose"
}

explicacoes = {
    "Arco elétrico": "Descarga de alta energia no óleo isolante.",
    "Falha térmica": "Sobreaquecimento detectado no enrolamento ou no isolamento.",
    "Descarga parcial": "Fenômeno elétrico que pode indicar degradação do isolamento.",
    "Degradação da celulose": "Degradação térmica dos componentes sólidos (papel isolante).",
    "Sem falha": "Transformador em condição de operação normal."
}

with st.form(key="form_diagnostico"):
    st.markdown('<div class="form-title">Preencha os dados dos gases (em ppm)</div>', unsafe_allow_html=True)

    gases = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    valores = []

    col1, col2 = st.columns(2)
    for i, gas in enumerate(gases):
        with (col1 if i < 3 else col2):
            val = st.number_input(f"{gas}", min_value=0.0, step=0.1, key=f"{gas}_input")
            valores.append(val)

    submit = st.form_submit_button("Diagnosticar")

if submit:
    entrada = np.array(valores).reshape(1, -1)
    entrada_normalizada = scaler.transform(entrada)
    pred = model.predict(entrada_normalizada)

    classe_id = int(pred[0])
    diagnostico = mapa_defeitos.get(classe_id, "Desconhecido")

    st.success(f"Tipo de falha diagnosticada: **{diagnostico}**")
    st.info(explicacoes.get(diagnostico, "Sem descrição disponível."))



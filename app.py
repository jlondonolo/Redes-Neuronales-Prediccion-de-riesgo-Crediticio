# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Riesgo Crediticio",
    page_icon="💰",
    layout="centered"
)

# Título y descripción
st.title("🏦 Predictor de Riesgo Crediticio")
st.markdown("Ingrese los datos del préstamo para evaluar el riesgo")

# Cargar modelo y preprocesadores
@st.cache_resource
def load_model():
    model = joblib.load('credit_risk_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_dict = joblib.load('label_encoders.joblib')
    return model, scaler, le_dict

model, scaler, le_dict = load_model()

# Crear el formulario
with st.form("loan_form"):
    # Información del préstamo
    st.subheader("📝 Información del Préstamo")
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amnt = st.number_input("Monto del Préstamo ($)", 
                                   min_value=1000, 
                                   max_value=40000, 
                                   value=10000)
        term = st.selectbox("Plazo", 
                           options=[36, 60],
                           format_func=lambda x: f"{x} meses")
        
    with col2:
        int_rate = st.number_input("Tasa de Interés (%)", 
                                  min_value=5.0, 
                                  max_value=30.0, 
                                  value=10.0)
        grade = st.selectbox("Calificación Crediticia", 
                           options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

    # Información personal
    st.subheader("👤 Información Personal")
    col3, col4 = st.columns(2)
    
    with col3:
        annual_inc = st.number_input("Ingreso Anual ($)", 
                                    min_value=20000, 
                                    max_value=200000, 
                                    value=50000)
        home_ownership = st.selectbox("Tipo de Vivienda",
                                    options=['RENT', 'OWN', 'MORTGAGE'])
        
    with col4:
        verification_status = st.selectbox("Estado de Verificación",
                                         options=['Verified', 'Source Verified', 'Not Verified'])
        purpose = st.selectbox("Propósito del Préstamo",
                             options=['debt_consolidation', 'credit_card', 'home_improvement',
                                    'other', 'major_purchase', 'small_business'])

    # Información crediticia
    st.subheader("💳 Información Crediticia")
    col5, col6 = st.columns(2)
    
    with col5:
        dti = st.number_input("Ratio Deuda-Ingreso (%)", 
                             min_value=0.0, 
                             max_value=50.0, 
                             value=15.0)
        delinq_2yrs = st.number_input("Incumplimientos (2 años)", 
                                     min_value=0, 
                                     max_value=10, 
                                     value=0)
        
    with col6:
        open_acc = st.number_input("Cuentas Abiertas", 
                                  min_value=0, 
                                  max_value=20, 
                                  value=3)
        revol_util = st.number_input("Utilización Crédito Rotativo (%)", 
                                    min_value=0.0, 
                                    max_value=100.0, 
                                    value=50.0)

    submitted = st.form_submit_button("Evaluar Riesgo")

# Procesar predicción
if submitted:
    try:
        # Preparar datos
        input_data = {
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'grade': grade,
            'home_ownership': home_ownership,
            'annual_inc': annual_inc,
            'verification_status': verification_status,
            'purpose': purpose,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'open_acc': open_acc,
            'revol_util': revol_util
        }

        # Crear DataFrame
        input_df = pd.DataFrame([input_data])

        # Procesar datos
        for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
            input_df[col] = le_dict[col].transform(input_df[col].astype(str))

        numeric_cols = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 
                       'dti', 'delinq_2yrs', 'open_acc', 'revol_util']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Hacer predicción
        risk_prob = float(model.predict(input_df)[0][0])
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📊 Resultados")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.metric("Probabilidad de Riesgo", f"{risk_prob:.1%}")
            
        with col8:
            risk_label = "Alto Riesgo" if risk_prob > 0.5 else "Bajo Riesgo"
            st.metric("Clasificación", risk_label)
            
        with col9:
            score = (1 - risk_prob) * 100
            st.metric("Score Crediticio", f"{score:.1f}")

        # Barra de riesgo
        st.progress(risk_prob)
        
        if risk_prob > 0.7:
            st.error("⚠️ Riesgo Alto: Se recomienda revisión detallada")
        elif risk_prob > 0.3:
            st.warning("⚠️ Riesgo Medio: Se sugiere evaluación adicional")
        else:
            st.success("✅ Riesgo Bajo: Perfil crediticio favorable")

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")

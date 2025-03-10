from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import joblib

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="API para evaluar el riesgo crediticio de préstamos",
    version="1.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://crediapp.onrender.com"],  # Dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Métodos HTTP permitidos
    allow_headers=["*"],  # Headers permitidos
)


@app.options("/{full_path:path}")
async def preflight_handler(full_path: str, request: Request):
    return JSONResponse(content=None, status_code=200)

# Cargar el modelo y los preprocessadores
print("Cargando modelo y preprocessadores...")
try:
    model = joblib.load('credit_risk_model.joblib')
    scaler = joblib.load('credit_risk_scaler.joblib')
    le_dict = joblib.load('credit_risk_le_dict.joblib')
    print("Modelo y preprocessadores cargados exitosamente")
except Exception as e:
    print(f"Error al cargar los archivos: {e}")
    raise

class LoanRequest(BaseModel):
    loan_amnt: float = Field(..., gt=0, description="Monto del préstamo")
    term: int = Field(..., ge=0, description="Plazo del préstamo en meses")
    int_rate: float = Field(..., ge=0, description="Tasa de interés")
    grade: str = Field(..., description="Calificación del préstamo")
    home_ownership: str = Field(..., description="Tipo de propiedad")
    annual_inc: float = Field(..., gt=0, description="Ingreso anual")
    verification_status: str = Field(..., description="Estado de verificación")
    purpose: str = Field(..., description="Propósito del préstamo")
    dti: float = Field(..., ge=0, description="Ratio deuda-ingreso")
    delinq_2yrs: int = Field(..., ge=0, description="Número de morosidades en 2 años")
    open_acc: int = Field(..., ge=0, description="Número de cuentas abiertas")
    revol_util: float = Field(..., ge=0, description="Utilización de crédito rotativo")

@app.post("/predict")
async def predict_loan(loan_data: LoanRequest):
    try:
        # 1. Preparar los datos en el formato correcto
        input_data = pd.DataFrame([loan_data.dict()])
        
        # 2. Procesar variables numéricas
        numeric_features = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 
                          'dti', 'delinq_2yrs', 'open_acc', 'revol_util']
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # 3. Procesar variables categóricas
        categorical_features = ['grade', 'home_ownership', 'verification_status', 'purpose']
        for col in categorical_features:
            try:
                input_data[col] = le_dict[col].transform(input_data[col].astype(str))
            except ValueError as e:
                valid_categories = list(le_dict[col].classes_)
                raise HTTPException(
                    status_code=400,
                    detail=f"Valor inválido para {col}. Valores permitidos: {valid_categories}"
                )
        
        # 4. Realizar predicción
        output = model.predict(input_data)
        
        # Verificar la forma de la salida
        if output.shape[1] == 1:  # Clasificación binaria
            probability = output[0][0]
        else:  # Clasificación multiclase
            probability = output[0][1]  # Cambiar el índice según la clase deseada
        
        prediction = "Alto Riesgo" if probability > 0.5 else "Bajo Riesgo"
        
        # 5. Preparar respuesta
        return {
            "prediccion": prediction,
            "probabilidad_riesgo": float(probability),
            "detalles": {
                "nivel_riesgo": "Alto" if probability > 0.7 else 
                               "Medio" if probability > 0.3 else "Bajo",
                "score_crediticio": int((1 - probability) * 100),
                "factores_riesgo": {
                    "dti_alto": loan_data.dti > 30,
                    "ingreso_bajo": loan_data.annual_inc < 30000,
                    "morosidades": loan_data.delinq_2yrs > 0
                }
            }
        }
        
    except IndexError as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    return {
        "status": "healthy",
        "modelo_cargado": model is not None,
        "preprocessadores_cargados": {
            "scaler": scaler is not None,
            "encoders": le_dict is not None
        }
    }

@app.get("/model-info")
async def model_info():
    """Obtener información sobre el modelo"""
    return {
        "caracteristicas_numericas": ['loan_amnt', 'term', 'int_rate', 'annual_inc', 
                                    'dti', 'delinq_2yrs', 'open_acc', 'revol_util'],
        "caracteristicas_categoricas": {
            feature: list(le_dict[feature].classes_)
            for feature in ['grade', 'home_ownership', 'verification_status', 'purpose']
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

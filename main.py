from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(
    title="House Price Prediction API",
    description="A FastAPI-powered machine learning API that serves house price predictions using a pre-trained regression model and scaled input features."
)

# Input schema
class HouseData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict_price(data: HouseData):
    input_data = np.array([[
        data.CRIM, data.ZN, data.INDUS, data.CHAS,
        data.NOX, data.RM, data.AGE, data.DIS,
        data.RAD, data.TAX, data.PTRATIO,
        data.B, data.LSTAT
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return {
        "predicted_house_price": round(float(prediction[0]), 2)
    }
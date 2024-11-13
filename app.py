from fastapi import FastAPI, HTTPException

import streamlit as st
import numpy as np
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

import joblib
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

from pydantic import BaseModel
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

def preprocessing(input_features: InputFeatures):
    dict_f = {
    'Year': input_features.Year,
    'Engine_Size': input_features.Engine_Size,
    'Mileage': input_features.Mileage,
    'Type_Accent': input_features.Type == 'Accent',
    'Type_Land Cruiser': input_features.Type == 'LandCruiser',
    'Make_Hyundai': input_features.Make == 'Hyundai',
    'Make_Mercedes': input_features.Make == 'Mercedes',
    'Options_Full': input_features.Options == 'Full',
    'Options_Standard': input_features.Options == 'Standard'
    }

    features_list = [dict_f[key] for key in sorted(dict_f)]


    scaled_features = scaler.transform([list(dict_f.values
    ())])

    return scaled_features
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}



st.title("Car Price Prediction")

year = st.number_input("Year", min_value=1980, max_value=2024, value=2020, step=1)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.5)
mileage = st.number_input("Mileage (km)", min_value=0.0, value=50000.0)
car_type = st.selectbox("Car Type", ['Accent', 'LandCruiser'])
make = st.selectbox("Make", ['Hyundai', 'Mercedes'])
options = st.selectbox("Options", ['Full', 'Standard'])

if st.button("Predict"):
    input_data = InputFeatures(
        Year=year,
        Engine_Size=engine_size,
        Mileage=mileage,
        Type=car_type,
        Make=make,
        Options=options
    )

    processed_data = preprocessing(input_data)
    prediction = model.predict(processed_data)

    st.success(f"Predicted Car Price: ${prediction[0]:,.2f}")

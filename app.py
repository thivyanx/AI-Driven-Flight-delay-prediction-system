import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb


# Load the trained model

model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")  


# Page Configuration

st.set_page_config(page_title="Flight Delay Prediction", page_icon="✈️", layout="centered")

# Page Title and Description

st.title("Flight Delay Prediction")
st.write("This app predicts whether a flight will be **delayed by 60+ minutes** based on your inputs.")

st.markdown("---")


# Sidebar

st.sidebar.header("Enter Flight Details")

DayOfWeek = st.sidebar.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
DepTime = st.sidebar.number_input("Departure Time (e.g., 1829)", min_value=0.0, max_value=2359.0, step=1.0)
Airline = st.sidebar.text_input("Airline", "Southwest Airlines Co.")
Origin = st.sidebar.text_input("Origin Airport Code", "IND")
Dest = st.sidebar.text_input("Destination Airport Code", "BWI")
month = st.sidebar.selectbox("Month", list(range(1, 13)))
day = st.sidebar.slider("Day of Month", 1, 31, 1)

# Prepare input for prediction
input_df = pd.DataFrame({
    'DayOfWeek': [DayOfWeek],
    'DepTime': [DepTime],
    'Airline': [Airline],
    'Origin': [Origin],
    'Dest': [Dest],
    'month': [month],
    'day': [day]
})

# Prediction Button

st.subheader("Flight Status Prediction")

if st.button("Predict"):
    # Encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Align with model's features
    model_columns = getattr(model, "feature_names_in_", [])
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Make prediction
    prediction = model.predict(input_encoded)
    result = "Delayed 60+ minutes" if prediction[0] == 1 else "On Time"

    # Show result
    st.success(f"Prediction Result: **{result}**")
    
# Footer
st.markdown("---")
st.caption("Developed by Thivyan K | Flight Delay Prediction Project")

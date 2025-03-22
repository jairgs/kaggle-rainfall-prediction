import streamlit as st
import pandas as pd
import numpy as np
#import pickle
from joblib import load

# Load your model
model = load("models/rainfall_classifier_v1.joblib")

# App title
st.title("🌧️ Rainfall Prediction")

st.write("The prediction is done with 2 ensambles (Random Forests and XGBoost trained on GPU) using Optuna studies to tune hyperparameters using Bayesian Optimization.")


# Links to Kaggle and GitHub
st.markdown("""
<p style='margin-top: -10px;'>
  <a href="https://www.kaggle.com/competitions/playground-series-s5e3" target="_blank" style="text-decoration: none; margin-right: 15px;">
    <img src="https://www.kaggle.com/static/images/site-logo.svg" alt="Kaggle" style="height: 20px; vertical-align: middle;"/> View Competition
  </a>
  <a href="https://github.com/jairgs/kaggle-rainfall-prediction" target="_blank" style="text-decoration: none;">
    <img src="https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png" alt="GitHub" style="height: 30px; vertical-align: middle; margin-right: 5px;"/> GitHub Repo
  </a>
</p>
""", unsafe_allow_html=True)


# Sidebar inputs
st.sidebar.header("Enter Weather Data")

def user_input_features():
    cloud = st.sidebar.slider("Cloud (%)", 0, 100, 50)
    day = st.sidebar.slider("Day of the Year", 1, 365, 1)
    dewpoint = st.sidebar.slider("Dew Point (°C)", -10.0, 30.0, 10.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    maxtemp = st.sidebar.slider("Max Temp (°C)", -10.0, 40.0, 30.0)
    mintemp = st.sidebar.slider("Min Temp (°C)", -10.0, 35.0, 15.0)
    pressure = st.sidebar.slider("Pressure (hPa)", 990.0, 1050.0, 1010.0)
    sunshine = st.sidebar.slider("Sunshine Hours", 0.0, 15.0, 7.5)
    temperature = st.sidebar.slider("Temperature (°C)", -10.0, 45.0, 22.0)
    winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
    windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 80.0, 15.0)

    data = {
        "cloud": cloud,
        "day": day,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "maxtemp": maxtemp,
        "mintemp": mintemp,
        "pressure": pressure,
        "sunshine": sunshine,
        "temperature": temperature,
        "winddirection": winddirection,
        "windspeed": windspeed
    }

    return pd.DataFrame([data])


# Get user input
input_df = user_input_features()

def add_derivative_columns(df):
     #from my SVM model
    df['week']=df.day//7
    df['winddirection_windspeed']=df.winddirection*df.windspeed
    df['dewpoint_week']=df.dewpoint*df.week

    #from other kaggle notebooks
    df['dewpoint_maxtemp']=df.maxtemp-df.dewpoint
    df['dewpoint_temp']=df.temperature-df.dewpoint
    df['sin_day']=np.sin(4 * np.pi * (df['day']-50) / 365)# 2 cycles in a year with 50 days shift
    df['temp_diff'] = df['maxtemp'] - df['mintemp']

    df=df[['cloud', 'day', 'dewpoint', 'dewpoint_maxtemp', 'dewpoint_temp',
       'dewpoint_week', 'humidity', 'maxtemp', 'mintemp', 'pressure',
       'sin_day', 'sunshine', 'temp_diff', 'temperature', 'week',
       'winddirection', 'winddirection_windspeed', 'windspeed']]
    return df

input_df=add_derivative_columns(input_df)

# Custom style for button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.75em 1.5em;
        font-size: 1.1em;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #cc3c3c;
        color: white;
    }
    div.stButton > button:first-child:focus {
        color: white !important;
        background-color: #cc3c3c !important;
    }
    </style>
""", unsafe_allow_html=True)



# Predict button
if st.button("Predict Rainfall Probability"):
    proba = model.predict_proba(input_df)[0, 1]
    st.markdown("### Predicted Probability of Rainfall")

    # Big bold centered number
    st.markdown(
        f"<h1 style='text-align: center; color: #1f77b4; font-size: 60px;'>{proba:.1%}</h1>",
        unsafe_allow_html=True
    )

    # Visual feedback bar
    st.progress(min(proba, 1.0))

    # Optional text insight
    if proba > 0.8:
        st.success("⚠️ High chance of rain — bring an umbrella!")
    elif proba > 0.5:
        st.info("🌥️ Moderate chance — be prepared!")
    else:
        st.info("🌤️ Low chance of rain — you're probably safe.")


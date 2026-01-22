import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="Sleep Disorder Prediction", layout="wide", page_icon="💤")
st.title("💤 Sleep Disorder Prediction - Multinomial Regression")

def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df.drop(columns=["Person ID", "Quality of Sleep"], inplace=True, errors='ignore')

    # Normalize BMI Category
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal", "Obese": "Overweight"})

    # Split Blood Pressure
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors='coerce')
        df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(columns=["Blood Pressure"], inplace=True)

    df.dropna(inplace=True)
    return df


def preprocess_for_model(df):
    df_model = df.copy()
    label_encoders = {}
    for col in df_model.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    return df_model, label_encoders

# Load dataset
df = load_data()
df['Sleep Disorder'] = df['Sleep Disorder'].replace("None", "Normal (no sleep disorder)")

# Model Training
df_model, encoders = preprocess_for_model(df)
X = df_model.drop(columns=["Sleep Disorder"])
y = df_model["Sleep Disorder"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class="multinomial", max_iter=1000)
model.fit(X_train, y_train)

# Predict user input
st.subheader("📋 Input Lifestyle & Health Data")
input_data = {}

with st.form("input_form"):
    input_data = {}
    for col in X.columns:
        if col in ["Age"]:
            input_data[col] = st.slider("Age", 10, 100, int(df[col].mean()))
        elif col in ["Gender", "Occupation", "BMI Category"]:
            options = list(encoders[col].classes_)
            input_data[col] = st.selectbox(col, options)
        elif col in ["Sleep Duration"]:
            input_data[col] = st.slider("Sleep Duration (hours)", 1, 24, int(df[col].mean()))
        elif col in ["Physical Activity Level"]:
            input_data[col] = st.slider("Physical Activity Level (minutes/day)", 30, 90, int(df[col].mean()))
        elif col in ["Stress Level"]:
            input_data[col] = st.slider("Stress Level (1-10)", 1, 10, int(df[col].mean()))
        elif col in ["Heart Rate"]:
            input_data[col] = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=int(df[col].mean()))
        elif col in ["Daily Steps"]:
            input_data[col] = st.number_input("Daily Steps", min_value=0, max_value=20000, value=int(df[col].mean()))
        elif col in ["BP_Systolic"]:
            input_data[col] = st.number_input("Blood Pressure Systolic", min_value=80, max_value=200, value=int(df[col].mean()))
        elif col in ["BP_Diastolic"]:   
            input_data[col] = st.number_input("Blood Pressure Diastolic", min_value=40, max_value=120, value=int(df[col].mean()))
        elif col in encoders:
            options = list(encoders[col].classes_)
            input_data[col] = st.selectbox(col, options)
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    submitted = st.form_submit_button("Predict Sleep Disorder")

if submitted:
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    pred = model.predict(input_df)[0]
    label = encoders['Sleep Disorder'].inverse_transform([pred])[0]

    st.subheader("😴 Predicted Sleep Disorder")
    st.success(f"Prediction: {label}")

    st.write("\n📈 Model accuracy on test data:")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred, target_names=encoders['Sleep Disorder'].classes_))

st.markdown("---")

st.caption("Model: Multinomial Logistic Regression | Data: Sleep Health and Lifestyle Dataset from Kaggle | Created by @serafiua")

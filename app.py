import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    layout="wide",
    page_icon="💤",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; background-color: #4CAF50; color: white;
        font-weight: bold; border-radius: 10px; height: 50px;
    }
    .stButton>button:hover { background-color: #45a049; }
    h1 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING & PROCESSING (CACHED) ---
@st.cache_data
def load_data():
    file_path = "Sleep_health_and_lifestyle_dataset.csv"
    
    if not os.path.exists(file_path):
        st.warning("⚠️ Dataset not found. Using dummy data.")
        data = {'Gender': ['Male']*100, 'Age': np.random.randint(20, 60, 100), 'Sleep Disorder': [np.nan]*50 + ['Insomnia']*25 + ['Sleep Apnea']*25, 'BMI Category': ['Normal']*100, 'Sleep Duration': [7]*100, 'Physical Activity Level': [50]*100, 'Stress Level': [5]*100, 'Heart Rate': [70]*100, 'Daily Steps': [5000]*100, 'BP_Systolic': [120]*100, 'BP_Diastolic': [80]*100}
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)

    # Langkah 1: Isi NaN di kolom Sleep Disorder dengan "Normal"
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna("Normal")
    
    # Langkah 2: Replace string "None" jadi "Normal" 
    df['Sleep Disorder'] = df['Sleep Disorder'].replace({"None": "Normal"})

    # Cleaning kolom lain
    df.drop(columns=["Person ID", "Quality of Sleep", "Occupation"], inplace=True, errors='ignore')
    
    # Logic BMI: Normal vs Overweight
    df["BMI Category"] = df["BMI Category"].replace({
        "Normal Weight": "Normal", 
        "Obese": "Overweight" 
    })
    
    # Split Blood Pressure
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors='coerce')
        df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(columns=["Blood Pressure"], inplace=True)

    df.dropna(inplace=True)
    
    # --- MANUAL ENCODING BMI ---
    bmi_map = {'Normal': 0, 'Overweight': 1}
    df['BMI_Code'] = df['BMI Category'].map(bmi_map)
    
    return df

@st.cache_resource
def train_model(df):
    df_model = df.copy()
    label_encoders = {}
    
    le_gender = LabelEncoder()
    df_model['Gender'] = le_gender.fit_transform(df_model['Gender'])
    label_encoders['Gender'] = le_gender
    
    le_target = LabelEncoder()
    df_model['Sleep Disorder'] = le_target.fit_transform(df_model['Sleep Disorder'])
    label_encoders['Sleep Disorder'] = le_target
        
    X = df_model.drop(columns=["Sleep Disorder", "BMI Category"]) 
    y = df_model["Sleep Disorder"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class="multinomial", max_iter=3000) 
    model.fit(X_train, y_train)
    
    return model, label_encoders, X.columns, X_test, y_test

# --- 4. MAIN APP ---

df = load_data()
model, encoders, feature_columns, X_test, y_test = train_model(df)

# Sidebar
with st.sidebar:
    st.title("About Project")
    st.info(
        """
        This app predicts the likelihood of sleep disorders based on lifestyle and health metrics using Multinomial Logistic Regression.
        """, icon="ℹ️"
    )
    st.warning(
        """
        **DISCLAIMER**
        
        This prediction **should not be taken as absolute medical advice**. The model is trained on a specific demographic dataset that may not fully represent your condition.
        
        Always consult a healthcare professional for your symptoms.
        """, icon="⚠️"
    )
    st.markdown("---")
    st.caption("Data Source: Kaggle Sleep Health Dataset")
    st.caption("Created by **serafiua**")

# Main page
st.title("💤 Sleep Disorder Predictor")
st.markdown("Enter your daily habits and health metrics below to get an AI-powered health assessment.")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Profile")
        age = st.number_input("Age", 10, 100, 30)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        bmi_options = ['Normal', 'Overweight']
        bmi = st.selectbox("BMI Category", bmi_options)

    with col2:
        st.subheader("🏃 Lifestyle")
        sleep_dur = st.slider("Sleep Duration (Hours)", 4.0, 10.0, 7.5, step=0.1)
        stress = st.slider("Stress Level (1-10)", 1, 10, 4)
        activity = st.slider("Physical Activity (min/day)", 0, 120, 60)
        steps = st.number_input("Daily Steps", 0, 20000, 7000, step=500)

    with col3:
        st.subheader("❤️ Vitals")
        bp_sys = st.number_input("BP Systolic", 90, 180, 120)
        bp_dia = st.number_input("BP Diastolic", 50, 120, 80)
        heart_rate = st.number_input("Heart Rate (bpm)", 50, 150, 70)

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Analyze Health Data")

if submitted:
    bmi_map_input = {'Normal': 0, 'Overweight': 1}
    bmi_val = bmi_map_input.get(bmi, 0) 

    input_data = {
        "Gender": encoders['Gender'].transform([gender])[0],
        "Age": age,
        "Sleep Duration": sleep_dur,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "BP_Systolic": bp_sys,
        "BP_Diastolic": bp_dia,
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "BMI_Code": bmi_val 
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    # Predict
    prediction_idx = model.predict(input_df)[0]
    prediction_label = encoders['Sleep Disorder'].inverse_transform([prediction_idx])[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    # UI Result
    st.markdown("### 📊 Prediction Result")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if "Normal" in prediction_label:
            st.success(f"**Healthy!** Prediction: **{prediction_label}**")
            st.markdown("Keep up your healthy lifestyle!")
        else:
            st.warning(f"⚠️ Indication: **{prediction_label}**")
            st.markdown("Based on the data, there might be signs of sleep disturbance.")
            
    with res_col2:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    with st.expander("🕵️ Debugging & Stats"):
        st.write(f"**Target Classes in Model:** {encoders['Sleep Disorder'].classes_}")
        
        # Grafik Probabilitas
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame(probs, index=encoders['Sleep Disorder'].classes_, columns=['Probability'])
        st.bar_chart(prob_df)
        
        st.markdown("---")
        
        # Classification Report Table
        st.write("📈 **Model Performance on Test Set:**")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=encoders['Sleep Disorder'].classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df.style.format("{:.2f}"))

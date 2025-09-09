import streamlit as st
import numpy as np
import pickle

# Load your trained ML model
model = pickle.load(open("model.pkl",rb))#ure this file exists

# --- THEME TOGGLE ---
theme = st.selectbox("🌗 Choose Theme", ["Light", "Dark"])

# --- HTML + CSS STYLING BASED ON THEME ---
def apply_custom_css(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: white;
            }
            .stApp {
                background-color: #121212;
                color: white;
            }
            input, select {
                background-color: #1e1e1e;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            body {
                background-color: #ffffff;
                color: black;
            }
            .stApp {
                background-color: #ffffff;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css(theme)

# --- TITLE ---
st.markdown(f"""
    <h2 style='text-align: center; color: {"white" if theme == "Dark" else "black"}'>
        ❤️ Heart Disease Prediction
    </h2>
""", unsafe_allow_html=True)

# --- FORM ---
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    
    submit = st.form_submit_button("Predict")

# --- PREDICTION ---
if submit:
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "True" else 0

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        # st.success("✅ No Heart Disease Detected")

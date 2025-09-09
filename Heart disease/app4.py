import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar Navigation (only Home and Data Visualization)
nav = st.sidebar.radio("Go to", ["Home", "Data Visualization"])

# If navigating manually, update session state
if nav != st.session_state.page and st.session_state.page != "Prediction Result":
    st.session_state.page = nav

# Function to go to result page
def go_to_result_page(prediction):
    st.session_state.prediction_result = prediction
    st.session_state.page = "Prediction Result"
    st.rerun()

# 🏠 HOME PAGE
if st.session_state.page == "Home":
    st.title('🫀 Heart Disease Prediction')
    st.markdown('Enter patient details below to predict the **risk of heart disease**.')

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Name", value="Hema")
        age = st.number_input("Enter Age", 20, 100, 52)
        sex = st.selectbox("Enter Sex", ["Male", "Female"])
        cp = st.selectbox("Enter Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Enter Resting Blood Pressure (in mm Hg)", 80, 200, 120)
        chol = st.number_input("Enter Serum Cholestoral in mg/dl", 100, 600, 240)
        fbs = st.selectbox("Enter Fasting Blood Sugar > 120 mg/dl", [0, 1])
    with col2:
        restcg = st.selectbox("Enter Resting Electrocardiographic results (0-2)", [0, 1, 2])
        thalach = st.number_input("Enter Maximum Heart Rate Achieved", 70, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Enter ST depression induced by exercise", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Enter slope of peak exercise ST segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Enter Number of major vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Enter Thal(1=Normal,2=Fixed,3=Reversible)", [1, 2, 3])

    if st.button("Predict"):
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restcg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)[0]
        go_to_result_page(prediction)

# 🧾 INTERNAL: PREDICTION RESULT PAGE (NOT IN SIDEBAR)
elif st.session_state.page == "Prediction Result":
    st.title("🧾 Prediction Result")

    if st.session_state.prediction_result is not None:
        if st.session_state.prediction_result == 1:
            st.warning('⚠️ High risk: Patient has heart disease.')
        else:
            st.success('✅ Low risk: Patient does not have heart disease.')
    else:
        st.info("⚠️ No prediction found. Please go to Home to make a prediction.")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

# 📊 DATA VISUALIZATION PAGE
elif st.session_state.page == "Data Visualization":
    st.title("📊 Heart Disease Data Visualization")

    df = pd.read_csv('heart.csv')
    st.write("#### Dataset Preview")
    st.dataframe(df.head())

    st.write("#### Age Distribution")
    fig_age, ax_age = plt.subplots()
    ax_age.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
    ax_age.set_xlabel('Age')
    ax_age.set_ylabel('Count')
    st.pyplot(fig_age)

    # Add more plots here if needed...

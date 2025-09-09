import streamlit as st
import pickle 
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open('model.pkl','rb'))

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Prediction", "Data Visualization"])

if page == "Prediction":
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
        oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Enter the slope of the peak exercise ST segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Enter Number of major vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Enter Thal(1=Normal,2=Fixed,3=Reversible)", [1, 2, 3])

    if st.button('Predict'):
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.warning('⚠️ High risk: Patient has heart disease.')
        else:
            st.success('✅ Low risk: Patient does not have heart disease.')

elif page == "Data Visualization":
    st.title("📊 Data Visualization Example")
    # Example: Show a random bar chart (replace with your own data)
    data = np.random.randint(10, 100, size=5)
    labels = ['A', 'B', 'C', 'D', 'E']
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_ylabel('Value')
    ax.set_title('Sample Bar Chart')
    st.pyplot(fig)
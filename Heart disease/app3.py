import streamlit as st
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the trained model
model = pickle.load(open('model.pkl','rb'))

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Prediction", "Data Visualization"])

if page == "Prediction":
    st.title('🫀 Heart Disease Prediction')
    st.markdown('Enter patient details below to predict the **risk of heart disease**.')

    col1, col2 = st.columns(2)

    # Define the DataFrame
    data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
    df = pd.DataFrame(data)

    # Now you can use df
    print(df.head())

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
    st.title("📊 Heart Disease Data Visualization")

    # Load your dataset (make sure heart.csv is in the same folder)
    df = pd.read_csv('heart.csv')

    st.write("####  Heart Disease Data", df.info())

    # Age distribution
    st.write("#### Age Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # Heart disease frequency by sex
    st.write("#### Heart Disease Frequency by Sex")
    sex_counts = df.groupby('sex')['target'].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(['Female', 'Male'], sex_counts)
    ax2.set_ylabel('Proportion with Heart Disease')
    st.pyplot(fig2)

        # Chest Pain Type (cp) distribution
    st.write("#### Chest Pain Type Distribution")
    fig_cp, ax_cp = plt.subplots()
    df['cp'].value_counts().sort_index().plot(kind='bar', ax=ax_cp)
    ax_cp.set_xlabel('Chest Pain Type')
    ax_cp.set_ylabel('Count')
    st.pyplot(fig_cp)

    # Resting Blood Pressure (trestbps) distribution
    st.write("#### Resting Blood Pressure Distribution")
    fig_trestbps, ax_trestbps = plt.subplots()
    ax_trestbps.hist(df['trestbps'], bins=20, color='orange', edgecolor='black')
    ax_trestbps.set_xlabel('Resting Blood Pressure (mm Hg)')
    ax_trestbps.set_ylabel('Count')
    st.pyplot(fig_trestbps)

    # Serum Cholesterol (chol) distribution
    st.write("#### Serum Cholesterol Distribution")
    fig_chol, ax_chol = plt.subplots()
    ax_chol.hist(df['chol'], bins=20, color='green', edgecolor='black')
    ax_chol.set_xlabel('Serum Cholesterol (mg/dl)')
    ax_chol.set_ylabel('Count')
    st.pyplot(fig_chol)

    # Fasting Blood Sugar (fbs) distribution
    st.write("#### Fasting Blood Sugar > 120 mg/dl Distribution")
    fig_fbs, ax_fbs = plt.subplots()
    df['fbs'].value_counts().sort_index().plot(kind='bar', ax=ax_fbs)
    ax_fbs.set_xlabel('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)')
    ax_fbs.set_ylabel('Count')
    st.pyplot(fig_fbs)

    # Resting ECG (restcg) distribution
    st.write("#### Resting ECG Distribution")
    fig_restcg, ax_restcg = plt.subplots()
    df['restecg'].value_counts().sort_index().plot(kind='bar', ax=ax_restcg)
    ax_restcg.set_xlabel('Resting ECG Result')
    ax_restcg.set_ylabel('Count')
    st.pyplot(fig_restcg)

    # Maximum Heart Rate Achieved (thalach) distribution
    st.write("#### Maximum Heart Rate Achieved Distribution")
    fig_thalach, ax_thalach = plt.subplots()
    ax_thalach.hist(df['thalach'], bins=20, color='purple', edgecolor='black')
    ax_thalach.set_xlabel('Maximum Heart Rate Achieved')
    ax_thalach.set_ylabel('Count')
    st.pyplot(fig_thalach)

    # Exercise Induced Angina (exang) distribution
    st.write("#### Exercise Induced Angina Distribution")
    fig_exang, ax_exang = plt.subplots()
    df['exang'].value_counts().sort_index().plot(kind='bar', ax=ax_exang)
    ax_exang.set_xlabel('Exercise Induced Angina (1=Yes, 0=No)')
    ax_exang.set_ylabel('Count')
    st.pyplot(fig_exang)

    # ST depression induced by exercise (oldpeak) distribution
    st.write("#### ST Depression (Oldpeak) Distribution")
    fig_oldpeak, ax_oldpeak = plt.subplots()
    ax_oldpeak.hist(df['oldpeak'], bins=20, color='red', edgecolor='black')
    ax_oldpeak.set_xlabel('ST Depression (Oldpeak)')
    ax_oldpeak.set_ylabel('Count')
    st.pyplot(fig_oldpeak)

    # Slope of the peak exercise ST segment (slope) distribution
    st.write("#### Slope of Peak Exercise ST Segment Distribution")
    fig_slope, ax_slope = plt.subplots()
    df['slope'].value_counts().sort_index().plot(kind='bar', ax=ax_slope)
    ax_slope.set_xlabel('Slope')
    ax_slope.set_ylabel('Count')
    st.pyplot(fig_slope)

    # Number of major vessels (ca) distribution
    st.write("#### Number of Major Vessels (ca) Distribution")
    fig_ca, ax_ca = plt.subplots()
    df['ca'].value_counts().sort_index().plot(kind='bar', ax=ax_ca)
    ax_ca.set_xlabel('Number of Major Vessels')
    ax_ca.set_ylabel('Count')
    st.pyplot(fig_ca)

    # Thalassemia (thal) distribution
    st.write("#### Thalassemia (thal) Distribution")
    fig_thal, ax_thal = plt.subplots()
    df['thal'].value_counts().sort_index().plot(kind='bar', ax=ax_thal)
    ax_thal.set_xlabel('Thalassemia (1=Normal, 2=Fixed, 3=Reversible)')
    ax_thal.set_ylabel('Count')
    st.pyplot(fig_thal)

    # Correlation heatmap
    st.write("#### Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
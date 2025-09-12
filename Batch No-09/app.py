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
    st.session_state.page = 'Prediction'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = {}

# Sidebar Navigation (only Home and Data Visualization)
nav = st.sidebar.selectbox("Go to", ["Prediction", "Data Visualization"])

# Sync nav selection with session_state (but prevent changing if on internal page)
if nav != st.session_state.page and st.session_state.page != "Prediction Result":
    st.session_state.page = nav

# Function to go to result page
def go_to_result_page(prediction, symptoms):
    st.session_state.prediction_result = prediction
    st.session_state.symptoms = symptoms
    st.session_state.page = "Prediction Result"
    st.rerun()

# 🏠 HOME PAGE
if st.session_state.page == "Prediction":
    st.title('🫀 Heart Disease Prediction')
    st.markdown('Enter patient details below to predict the **risk of heart disease**.')

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Name", value="Hema")
        age = st.number_input("Enter Age", 20, 100, 52)
        sex = st.selectbox("Enter Sex", ["Male", "Female"])
        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal pain", "Asymptomatic"]
        cp = st.selectbox("Enter Chest Pain Type", cp_options)
        cp_val = cp_options.index(cp)
        trestbps = st.number_input("Enter Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Enter Serum Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    with col2:
        restcg = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (1=Normal,2=Fixed,3=Reversible)", [1, 2, 3])

    if st.button("Predict"):
        sex_val = 1 if sex == "Male" else 0
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_data = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)
        prediction = model.predict(input_data)[0]

        # Prepare symptoms dictionary
        symptoms = {
            "Age": age,
            "Sex (Male=1)": sex_val,
            "Chest Pain Type": cp_val,
            "Resting BP": trestbps,
            "Cholesterol": chol,
            "Fasting Sugar >120": fbs,
            "Resting ECG": restcg,
            "Max Heart Rate": thalach,
            "Exercise Angina": exang,
            "ST Depression": oldpeak,
            "Slope": slope,
            "Major Vessels": ca,
            "Thalassemia": thal
        }

        go_to_result_page(prediction, symptoms)

# 🧾 PREDICTION RESULT PAGE
elif st.session_state.page == "Prediction Result":
    st.title("🧾 Prediction Result")

    if st.session_state.prediction_result is not None:
        if st.session_state.prediction_result == 1:
            st.warning('⚠️ High risk: Patient has heart disease.')
        else:
            st.success('✅ Low risk: Patient does not have heart disease.')

        # Chart of entered symptoms
        if "symptoms" in st.session_state:
            st.write("### Patient Symptoms Overview")
            symptom_df = pd.DataFrame({
                'Symptom': list(st.session_state.symptoms.keys()),
                'Value': list(st.session_state.symptoms.values())
            })

            # Allow user to select which symptom to visualize
            selected_symptom = st.selectbox("Select Symptom to Highlight in Graph", list(symptom_df['Symptom']))

            # Plot all symptoms in one horizontal bar chart, highlight selected
            colors = ['teal' if s != selected_symptom else 'orange' for s in symptom_df['Symptom']]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(symptom_df['Symptom'], symptom_df['Value'], color=colors)
            ax.set_xlabel("Value")
            ax.set_title(f"Entered Patient Symptom Values (Highlight: {selected_symptom})")
            # Define custom scales for each symptom
            scales = {
                "Age": (20, 100),
                "Sex (Male=1)": (0, 1),
                "Chest Pain Type": (0, 3),
                "Resting BP": (80, 200),
                "Cholesterol": (100, 600),
                "Fasting Sugar >120": (0, 1),
                "Resting ECG": (0, 2),
                "Max Heart Rate": (70, 220),
                "Exercise Angina": (0, 1),
                "ST Depression": (0.0, 10.0),
                "Slope": (0, 2),
                "Major Vessels": (0, 3),
                "Thalassemia": (1, 3)
            }
            ax.set_xlim(min([v[0] if isinstance(v, tuple) and len(v) == 2 else 0 for v in scales.values()]),
                        max([v[1] if isinstance(v, tuple) and len(v) == 2 else 1 for v in scales.values()]))
            st.pyplot(fig)
    else:
        st.info("⚠️ No prediction found. Please go to Home to make a prediction.")

    if st.button("🔙 Back to Prediction"):
        st.session_state.page = "Prediction"
        st.rerun()

# 📊 DATA VISUALIZATION PAGE
elif st.session_state.page == "Data Visualization":
    st.title("📊 Heart Disease Data Visualization")

    df = pd.read_csv('heart.csv')
    st.write("#### Dataset Preview")
    st.dataframe(df.head())

    st.write("#### Age Distribution")
    fig_age, ax_age = plt.subplots()
    ax_age.hist(df['age'], bins=20, color='blue', edgecolor='black')
    ax_age.set_xlabel('Age')
    ax_age.set_ylabel('Count')
    st.pyplot(fig_age)
    
    # Heart disease frequency by sex
    st.write("#### Frequency by Sex")
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

    # Target variable distribution
    st.write("#### Target Distribution (Heart Disease)")
    fig_target, ax_target = plt.subplots()
    df['target'].value_counts().sort_index().plot(kind='bar', ax=ax_target, color=['salmon', 'lightgreen'])
    ax_target.set_xlabel('Target (0=No Disease, 1=Disease)')
    ax_target.set_ylabel('Count')
    ax_target.set_title('Heart Disease Target Distribution')
    st.pyplot(fig_target)

    # If you want to display an image, use a valid filename that exists in your project folder.
    # For example:
    # st.image("your_image.png")  # Make sure 'your_image.png' exists in your folder
    # Remove or update any references to missing images.

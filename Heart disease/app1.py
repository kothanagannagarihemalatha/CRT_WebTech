import streamlit as st
import pickle 
import numpy as np


#load the saved model & Scaler

model = pickle.load(open('model.pkl','rb'))

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
#UI (app)
st.title('🫀 Heart Disease Prediction')

st.markdown(f'''
        <h1style='text-align: center; color: {"Black"if theme=='Dark'else'black'};'>
Enter patient details below to predict the **risk of heart disease**.
    </h1
''', unsafe_allow_html=True)
#take input features
col1,col2 = st.columns(2)

with col1:
 age = st.number_input("Enter Age", 20,100,52)
 sex = st.selectbox("Enter Sex", [0,1] )#Male=1, Female=0
 cp = st.selectbox("Enter Chest Pain Type (0-3)",[0,1,2,3])
 trestbps = st.number_input("Enter Resting Blood Pressure (in mm Hg)", 80,200,120)
 chol = st.number_input("Enter Serum Cholestoral in mg/dl", 100,600,240)
 fbs = st.selectbox("Enter Fasting Blood Sugar > 120 mg/dl", [0,1]) #1 = True; 0=False

with col2:
 restcg=st.selectbox("Enter Resting Electrocardiographic esults (0-2)", [0,1,2])
 thalach= st.number_input("Enter Maximum Heart Rate Achieved", 70,220,150)
 exang= st.selectbox("Exercise Induced Angina", [0,1]) #yes=1, No=0
 oldpeak= st.number_input("Enter ST depression induced by exercise relative to rest", 0.0,10.0,1.0, step=0.1)
 slope= st.selectbox("Enter the slope of the peak exercise ST segment (0-2)",[0,1,2])
 ca= st.selectbox("Enter Number of major vessels (0-3) ",[0,1,2,3])
 thal=st.selectbox("Enter Thal(1=Normal,2=Fixed,3=Reversible)",[1,2,3])
      
if st.button('Predict'):
 #Arrange features as in training
 input_data=np.array([[age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]])

 #prediction
 prediction=model.predict(input_data)[0]
 
 if prediction==1:
  st.warning('⚠️ High risk: Patient has heart disease.')
 else:
  st.success('✅ Low risk: Patient does not have heart disease.')
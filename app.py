import streamlit as st
import pandas as pd
import joblib
model=joblib.load('knn_heart.pkl')
scaler=joblib.load('scaler.pkl')
columns=joblib.load('columns.pkl')
st.title('Heart Disease Prediction')
st.markdown("Provide Following Details")

age=st.slider('Age',18,100,40)
sex=st.selectbox('Sex',['M','F'])
cp=st.selectbox('Cp Type',['ATA','NAP','ASY','TA'])
restbps=st.number_input('Resting Blood Pressure',80,200,120)
chol=st.number_input('Cholesterol',100,600,200)
fbs=st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1])
restecg=st.selectbox('Resting Electrocardiographic Results',['Normal','ST','LVH'])
max_hr=st.number_input('Maximum Heart Rate Achieved',70,200,150)
exang=st.selectbox('Exercise Induced Angina',[0,1])
oldpeak=st.number_input('ST Depression Induced by Exercise Relative to Rest',0.0,6.0,1.0)
slope=st.selectbox('Slope of the Peak Exercise ST Segment',['Up','Flat','Down'])

if st.button('Predict'):
    data={'Age':age,'Sex':sex,'ChestPainType':cp,'RestingBP':restbps,'Cholesterol':chol,'FastingBS':fbs,'RestingECG':restecg,'MaxHR':max_hr,'ExerciseAngina':exang,'Oldpeak':oldpeak,'ST_Slope':slope}
    df=pd.DataFrame(data,index=[0])
    for col in columns:
        if col not in df.columns:
            df[col]=0
    df=df[columns]
    df=scaler.transform(df)
    result=model.predict(df)[0]
    if result==0:
        st.success('No Heart Disease')
    else:
        st.error('Heart Disease')
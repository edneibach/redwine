import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model('wine')

def predict(dataframe):
    return predict_model(model,data=dataframe)

st.title("Red Wine Quality Predictor")

st.write("Kaggle URL for the original Dataset: \n \n https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009")

mycolumns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

fixacid = st.slider('Fixed Acidity', min_value=4.6, max_value=15.9)
volacid = st.slider("Volatile Acidity", min_value=0.12, max_value=1.58)
citacid = st.slider("Citric Acid", min_value=0.00, max_value=1.00)
ressugar = st.slider("Citric Acid", min_value=2.53, max_value=15.5)
chlorides = st.slider("Chlorides", min_value=0.00, max_value=0.61)
fsulfur = st.slider("Free Sulfur Dioxide", min_value=1, max_value=72)
tsulfur = st.slider("Total Sulfur Dioxide", min_value=6, max_value=289)
density = st.slider("Density", min_value=0.99, max_value=1.01)
ph = st.slider("pH", min_value=2.74, max_value=4.01)
sulph = st.slider("Sulphates", min_value=2.74, max_value=4.01)
alcohol = st.slider("Alcohol Percentage", min_value=8.4, max_value=14.9)

mybutton = st.button("What is the quality of this wine?")

if mybutton:
    df = pd.DataFrame(columns=mycolumns)
    df.loc[1] = [fixacid,volacid,citacid,ressugar,chlorides,fsulfur,tsulfur,density,ph,sulph,alcohol]
    st.write("This wine would be graded ", predict(df)['Label'][1])

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

import joblib
import requests

# response = requests.get('https://drive.google.com/drive/folders/1ufzjPwwq2Cb8zDxZskFisl6_gRqA9A7Q?usp=sharing') 

import urllib3 
content=urllib3.urlopen("https://drive.google.com/file/d/1yK_upqSxk0VBPbf802nkmJXbO7JOasZ2/view?usp=sharin") 



st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Type your semptom we will diagnsis you !!!")
st.text("what are your semptoms ?")
st.text("White all your symptoms")

@st.cache(allow_output_mutation = True)

def load_model():
    model = joblib.load('bert_qa_custom.joblib')
    return model

with st.spinner('loading Model into Memory...'):
    model = load_model()

text  = st.text_input('Enter yours symptoms here...')

if text:
    st.write("Response :")
    with st.spinner ('Searching for diagnsis...'):
        prediction = model.predict(text)
        st.write('answer:{}'.format(prediction[0]))
        st.write('title:{}'.format(prediction[1]))
        st.write('paragraph:{}'.format(prediction[2]))
    st.write("")


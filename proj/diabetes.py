import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#from sklearn.linear_model import LogisticRegression
#from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
#for title
st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Diabetes detection using ML</p>', unsafe_allow_html=True)
#img=Image.open('C:/Users/susil/Desktop/proj/diab/dia.png')
#st.image(img,caption='ml',use_column_width=True)
#get data
df=pd.read_csv('db.csv',encoding = "ISO-8859-1")
#subheader
st.subheader('Data Information')
#show data as table
st.dataframe(df)
#show statistics
st.write(df.describe())
#chart
chart=st.bar_chart(df)
#split data
x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values
#split for train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#get the feature input
def get_user_inp():
    Pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    Glucose=st.sidebar.slider('Glucose',0,199,117)
    BloodPressure=st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness=st.sidebar.slider('SkinThickness',0,99,33)
    Insulin=st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
    DiabetesPedigreeFunction=st.sidebar.slider('DiabetesPedigreeFunction',0.078,2.42,0.3725)
    Age=st.sidebar.slider('Age',21,81,29)
#store a dict into a variable
    user_inp={'Pregnancies':Pregnancies,
               'Glucose':Glucose,
                'BloodPressure':BloodPressure,
                'SkinThickness':SkinThickness,
                'Insulin':Insulin,
                'BMI':BMI,
                'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                'Age':Age }
#data to data frame
    features=pd.DataFrame(user_inp,index=[0])
    return features
#store the user input into a var
user_inp=get_user_inp()
#set a subheader and display the users input
st.subheader('User Input')
st.write(user_inp)
#train the model
RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
#show model's accuracy
st.subheader('Model Test Accuracy Score')
st.write(str(accuracy_score(y_test,RFC.predict(x_test))*100)+'%')
#store the model output in a variable
prediction=RFC.predict(user_inp)
st.subheader('Classifier ')
st.write(prediction)

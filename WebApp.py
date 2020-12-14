# Description

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from PIL import Image
import streamlit as st

# create a title and sub-title
st.write("""
# Diabetes detection
Detect if someone has diabetes using machine learning and python 1
""")
# Open and Dispaly an image
image = Image.open("diabetes detrectweb appp.png")
st.image(image, caption="ML", use_column_width=True)
# Get The Data
df = pd.read_csv("diabetes.csv")

# Set A Subheader
st.subheader('Data Information')
# Show the data has a table
st.dataframe(df)
# Show static
st.write(df.describe())
# to show chart
chart = st.bar_chart(df)

# Split the data into independent'x'and dependent 'y' variable
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
#split into 75#training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Get user input
def get_user_input():
    pregnancies = st.sidebar.slider("pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("glucose", 0, 199, 117)
    blood_Pressure = st.sidebar.slider("blood_pressure", 0, 122, 71)
    skin_Thickness = st.sidebar.slider("skin_thickness", 0, 99, 23)
    insulin = st.sidebar.slider("insulin", 0.0, 846.0, 30.0)
    BMI =st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF", 0, 17, 3)
    age = st.sidebar.slider("age", 21, 81, 29)

    #store
    user_data = {'pregnancies' : pregnancies,
                 'glucose' : glucose,
                 "blood_pressure" : blood_Pressure,
                 "skin_Thickness" : skin_Thickness,
                 "insulin" : insulin,
                 'BMI' : BMI,
                 'DPF' : DPF,
                 "age" : age
                 }
    # Transform the data  into a data frame
    features = pd.DataFrame(user_data, index= [0])
    return features

# store the user input into variable
user_input = get_user_input()

# set a Subheader and display  the users input
st.subheader('User Input: ')
st.write(user_input)

# Create and train  the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

# Show the matrics model
st.subheaders(' Model Test Accuracy score: ')
st.write( str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

# Store the modekl predxictions
prediction = RandomForestClassifier(user_input)

# sert a subheader(
st.subheader('classification: ')
st.write(prediction)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the label encoder
le = pickle.load(open('label_encoder_gender.pkl', 'rb'))

# Load the one hot encoder
ohe = pickle.load(open('onehot_encoder_geo.pkl', 'rb'))

# Set the title of the app
st.title('Customer Churn Prediction')

# Create input fields for the features
Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age')
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10, 1)
numofproducts = st.slider('Number of Products', 1, 4, 1)
hascard = st.radio('Has Card', [0, 1])
isactivemember = st.radio('Is Active Member', [0, 1])

input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [le.transform([Gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [numofproducts],
        'HasCrCard': [hascard],
        'IsActiveMember': [isactivemember],
        'EstimatedSalary': [estimated_salary],
    }
)

geo_encoded = ohe.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

predict_proba = prediction[0][0]

# Create a button to make predictions
if st.button('Predict'):    
    # Display the prediction
    if predict_proba > 0.5:
        st.write('The customer is likely to be churned')
    else:
        st.write('The customer is not likely to be churned')
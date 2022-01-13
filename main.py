import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Importing pickle files which are created in the jupyter notebook

df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
location = pickle.load(open('location.pkl', 'rb'))


st.title('Real Estate Price Predictor')

# Getting the data from user
# location
location = st.selectbox('Location', location)

# bath
bath = st.selectbox('No. of Baths', df['Baths'].unique())

# sqft
sqft = st.number_input('Total sqft')

# bhk
bhk = st.selectbox('bhk', df['bhk'].unique())

#inputs = [[location, bath, sqft, bhk]]

if st.button('Predict Price'):

    def price_predict(location, sqft, bath, bhk):
        x = np.zeros(len(df.columns))
        x[0] = bath
        x[1] = sqft
        x[2] = bhk
        if location != 'other':
            x[np.where(df.columns == location)[0][0]] = 1
        return model.predict([x])[0]

    result = price_predict(location, sqft, bath, bhk)

    st.title("The predicted price in Lakhs is {}".format(result))
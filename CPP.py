import streamlit as st
import pickle as pk
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Streamlit app title
st.title("Car Price Prediction")

# Load your data
df = pd.read_csv('D:\MSC DSAI\Python\Car Price - Car Price.csv')

# Load the pickled model
with open(r'D:\MSC DSAI\Python\xgb_model.pkl', 'rb') as file:
    xgb_model = pk.load(file)

# Initialize LabelEncoders for categorical columns
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_owner = LabelEncoder()
le_Seller_Type = LabelEncoder()

# Fit the encoders on your dataset
df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Model'] = le_model.fit_transform(df['Model'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])
df['Seller_Type'] = le_Seller_Type.fit_transform(df['Seller_Type'])
df['Transmission'] = le_transmission.fit_transform(df['Transmission'])
df['Owner'] = le_owner.fit_transform(df['Owner'])

# Dropdown inputs for categorical variables
brand = st.selectbox("Select Brand", le_brand.classes_)
model = st.selectbox("Select Model", le_model.classes_)
fuel = st.selectbox("Select Fuel Type", le_fuel.classes_)
Seller_Type = st.selectbox("Select Seller Type", le_Seller_Type.classes_)
transmission = st.selectbox("Select Transmission Type", le_transmission.classes_)
owner = st.selectbox("Select Owner Type", le_owner.classes_)

# Number inputs for numerical variables
year = st.number_input("Year of the car:", min_value=1900, max_value=2025, value=2010)
km_driven = st.number_input("Kilometers Driven:", min_value=0, max_value=1000000, value=50000)

# Encode categorical values
brand_encoded = le_brand.transform([brand])[0]
model_encoded = le_model.transform([model])[0]
fuel_encoded = le_fuel.transform([fuel])[0]
transmission_encoded = le_transmission.transform([transmission])[0]
owner_encoded = le_owner.transform([owner])[0]
Seller_Type_encoded = le_Seller_Type.transform([Seller_Type])[0]

# Create a dataframe for the input data
input_data = pd.DataFrame({
    'Brand': [brand_encoded],
    'Model': [model_encoded],
    'Year': [year],
    'KM_Driven': [km_driven],
    'Fuel': [fuel_encoded],
    'Seller_Type':[Seller_Type_encoded],
    'Transmission': [transmission_encoded],
    'Owner': [owner_encoded]
})

# Display the input data for review
st.write("Input data for prediction:", input_data)

# Make prediction when the user clicks the button
if st.button("Predict Car Price"):
    prediction = xgb_model.predict(input_data)
    st.success(f"Predicted Car Price: ₹ {prediction[0]:,.2f}")


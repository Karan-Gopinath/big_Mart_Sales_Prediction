import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load the trained model
try:
    with open('bigmart_model.pkl', 'rb') as file:
        model = pickle.load(file)
    if not isinstance(model, XGBRegressor):
        st.error("Loaded model is not an XGBRegressor. Please ensure 'bigmart_model.pkl' contains a trained XGBRegressor model.")
        st.stop()
except FileNotFoundError:
    st.error("Model file 'bigmart_model.pkl' not found. Please ensure it is in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the dataset for encoding references
try:
    data = pd.read_csv('Train.csv')
except FileNotFoundError:
    st.error("Dataset 'Train.csv' not found. Please ensure it is in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Handle missing values in dataset (avoid inplace=True)
data = data.assign(
    Item_Weight=data['Item_Weight'].fillna(data['Item_Weight'].mean()),
    Outlet_Size=data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])
)

# Replace inconsistent values in Item_Fat_Content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
)

# Initialize LabelEncoders for categorical columns
encoders = {}
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                      'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for column in categorical_columns:
    encoders[column] = LabelEncoder()
    encoders[column].fit(data[column])

# Streamlit app
st.title("Big Mart Sales Prediction")
st.write("Enter the item and outlet details to predict sales:")

# Input fields
item_identifier = st.selectbox("Item Identifier", data['Item_Identifier'].unique())
item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=10.0)
item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.0)
item_type = st.selectbox("Item Type", data['Item_Type'].unique())
item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=300.0, value=100.0)
outlet_identifier = st.selectbox("Outlet Identifier", data['Outlet_Identifier'].unique())
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=2025, value=2000)
outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
outlet_location_type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
outlet_type = st.selectbox("Outlet Type", data['Outlet_Type'].unique())

# Prepare input data
input_data = pd.DataFrame({
    'Item_Identifier': [item_identifier],
    'Item_Weight': [item_weight],
    'Item_Fat_Content': [item_fat_content],
    'Item_Visibility': [item_visibility],
    'Item_Type': [item_type],
    'Item_MRP': [item_mrp],
    'Outlet_Identifier': [outlet_identifier],
    'Outlet_Establishment_Year': [outlet_establishment_year],
    'Outlet_Size': [outlet_size],
    'Outlet_Location_Type': [outlet_location_type],
    'Outlet_Type': [outlet_type]
})

# Encode categorical features
try:
    for column in categorical_columns:
        input_data[column] = encoders[column].transform(input_data[column])
except Exception as e:
    st.error(f"Error encoding input data: {e}")
    st.stop()

# Predict
if st.button("Predict Sales"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Item Outlet Sales: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

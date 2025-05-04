import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle

# Load the dataset
big_mart_data = pd.read_csv('Train.csv')

# Handle missing values
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
big_mart_data['Outlet_Size'].fillna(big_mart_data['Outlet_Size'].mode()[0], inplace=True)

# Replace inconsistent values in Item_Fat_Content
big_mart_data['Item_Fat_Content'] = big_mart_data['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# Encode categorical features
encoder = LabelEncoder()
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                      'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for column in categorical_columns:
    big_mart_data[column] = encoder.fit_transform(big_mart_data[column])

# Split features and target
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train XGBoost Regressor
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Evaluate model
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('Training R Squared value = ', r2_train)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('Test R Squared value = ', r2_test)

# Save the trained model
with open('bigmart_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

print("Model saved as bigmart_model.pkl")
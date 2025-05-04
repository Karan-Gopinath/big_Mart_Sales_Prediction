# Big Mart Sales Prediction

This project predicts sales for Big Mart outlets using an XGBoost Regressor model. The model is trained on the Big Mart Sales dataset from Kaggle and deployed as a web app using Streamlit. The trained model is stored on Google Drive and downloaded at runtime for predictions.

## Live Demo

Try the deployed app: [Big Mart Sales Prediction on Streamlit](https://bigmartsalesprediction-xrztn9uzxy9ykhjjwjkvfh.streamlit.app/)

## Dataset

The dataset is sourced from Kaggle: [Big Mart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?resource=download&select=Train.csv). It contains features such as:

- `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`
- `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
- Target: `Item_Outlet_Sales`

Download `Train.csv` from the Kaggle link and place it in the project directory.

## Model

- **Algorithm**: XGBoost Regressor
- **Training R²**: ~0.876
- **Test R²**: ~0.502
- The trained model is saved as `bigmart_model.pkl`, uploaded to Google Drive, and downloaded by the Streamlit app for predictions.

## Project Structure

- `Train.csv`: Dataset (download from Kaggle).
- `big_mart_sales_prediction.py`: Script to preprocess data, train the XGBoost model, and upload it to Google Drive as `bigmart_model.pkl`.
- `app.py`: Streamlit app that downloads the model from Google Drive and predicts sales.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes unnecessary files (e.g., `bigmart_model.pkl`, `credentials.json`, `token.json`).
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`:
  - `numpy`, `pandas`, `scikit-learn`, `xgboost`, `streamlit`
  - `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, `google-api-python-client` (for Google Drive API)
  - `gdown` (for downloading the model)

## Setup

1. **Clone the Repository**:

   ```bash
# Big Mart Sales Prediction

This project predicts sales for Big Mart outlets using an XGBoost Regressor model. The model is trained on the Big Mart Sales dataset from Kaggle and deployed as a web app using Streamlit. The trained model is stored on Google Drive and downloaded at runtime for predictions.

## Live Demo

Try the deployed app: [Big Mart Sales Prediction on Streamlit](https://bigmartsalesprediction-xrztn9uzxy9ykhjjwjkvfh.streamlit.app/)

## Dataset

The dataset is sourced from Kaggle: [Big Mart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?resource=download&select=Train.csv). It contains features such as:

- `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`
- `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
- Target: `Item_Outlet_Sales`

Download `Train.csv` from the Kaggle link and place it in the project directory.

## Model

- **Algorithm**: XGBoost Regressor
- **Training R²**: ~0.876
- **Test R²**: ~0.502
- The trained model is saved as `bigmart_model.pkl`, uploaded to Google Drive, and downloaded by the Streamlit app for predictions.

## Project Structure

- `Train.csv`: Dataset (download from Kaggle).
- `big_mart_sales_prediction.py`: Script to preprocess data, train the XGBoost model, and upload it to Google Drive as `bigmart_model.pkl`.
- `app.py`: Streamlit app that downloads the model from Google Drive and predicts sales.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes unnecessary files (e.g., `bigmart_model.pkl`, `credentials.json`, `token.json`).
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`:
  - `numpy`, `pandas`, `scikit-learn`, `xgboost`, `streamlit`
  - `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, `google-api-python-client` (for Google Drive API)
  - `gdown` (for downloading the model)

## Setup

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:Karan-Gopinath/big_Mart_Sales_Prediction.git
   cd big_mart_sales_prediction   cd big_mart_sales_prediction

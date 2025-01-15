import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Global variable to store model columns
model_columns = None

def preprocess_data(data, fit_columns=None):
    required_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                        'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                        'casual', 'registered', 'cnt']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    objcols = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                    'workingday', 'weathersit']]
    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                                                     'workingday', 'weathersit'])
    numcols = data[['temp', 'atemp', 'hum', 'windspeed',
                    'casual', 'registered', 'cnt']]
    final_data = pd.concat([numcols, objcols_dummy], axis=1)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in final_data.columns:
                final_data[col] = 0  # Add missing columns with default value 0
        final_data = final_data[fit_columns]  # Reorder to match fit columns

    return final_data

# Define the Streamlit interface
st.title('Bike Count Prediction Model')

st.sidebar.header("Model Evaluation")

# File uploader for training data
# Declare the global variable earlier to avoid SyntaxError

train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv"])
if train_file:
    global model_columns
    train_data = pd.read_csv(train_file)
    st.write("### Training Data Loaded")
    st.dataframe(train_data.head())

    try:
        # Preprocess training data
        final_train_data = preprocess_data(train_data)
        model_columns = final_train_data.columns  # Save the training columns for reuse
        y_train = final_train_data['cnt']
        X_train = final_train_data.drop(['cnt', 'atemp', 'registered'], axis=1)

        # Train the model
        model = LinearRegression().fit(X_train, y_train)

        # Calculate metrics
        r_square = model.score(X_train, y_train)
        y_train_pred = model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        st.write(f"### R-Square on Training Data: {r_square:.4f}")
        st.write(f"### RMSE on Training Data: {rmse_train:.4f}")
    except KeyError as e:
        st.error(f"Error in processing training data: {e}")

# File uploader for testing data
st.sidebar.header("Upload Test Data")
test_file = st.sidebar.file_uploader("Upload Test Data", type=["csv"])
if test_file:
    test_data = pd.read_csv(test_file)
    st.write("### Test Data Loaded")
    st.dataframe(test_data.head())

    try:
        # Preprocess test data using the same columns as training
        final_test_data = preprocess_data(test_data, fit_columns=model_columns)
        y_test = final_test_data['cnt']
        X_test = final_test_data.drop(['cnt', 'atemp', 'registered'], axis=1)

        y_test_pred = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        st.write(f"### RMSE on Test Data: {rmse_test:.4f}")
    except KeyError as e:
        st.error(f"Error in processing test data: {e}")

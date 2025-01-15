import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# App title
st.title("Model Training and Evaluation App")

# Upload buttons for training and testing data
train_file = st.file_uploader("Upload Training Data (CSV):", type="csv")
test_file = st.file_uploader("Upload Test Data (CSV):", type="csv")

def preprocess_data(data):
    """
    Preprocess data by converting non-numeric columns to numeric and handling dates.
    """
    # Identify non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object', 'datetime']).columns

    # Convert non-numeric columns to numeric using one-hot encoding
    data = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)

    return data

if train_file:
    # Load training data
    st.write("### Training Data")
    train_data = pd.read_csv(train_file)
    st.write(train_data.head())

    # Extract X and y from training data
    if 'cnt' in train_data.columns:
        y_train = train_data['cnt']
        X_train = train_data.drop(['cnt'], axis=1)

        # Preprocess the data
        X_train = preprocess_data(X_train)

        # Drop multicollinear columns if present
        if 'atemp' in X_train.columns and 'registered' in X_train.columns:
            X_train = X_train.drop(['atemp', 'registered'], axis=1)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Compute R² score
        r_square = model.score(X_train, y_train)

        # Compute RMSE on training data
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Display R² and RMSE
        st.write(f"#### R² Score (Training Data): {r_square:.4f}")
        st.write(f"#### RMSE (Training Data): {train_rmse:.4f}")
    else:
        st.error("Training data must include a column named 'cnt' as the dependent variable.")

if test_file:
    # Load test data
    st.write("### Test Data")
    test_data = pd.read_csv(test_file)
    st.write(test_data.head())

    # Extract X and y from test data
    if 'cnt' in test_data.columns:
        y_test = test_data['cnt']
        X_test = test_data.drop(['cnt'], axis=1)

        # Preprocess the data
        X_test = preprocess_data(X_test)

        # Ensure same features as training data
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Predict on test data
        y_test_pred = model.predict(X_test)

        # Compute RMSE on test data
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Display RMSE
        st.write(f"#### RMSE (Test Data): {test_rmse:.4f}")
    else:
        st.error("Test data must include a column named 'cnt' as the dependent variable.")

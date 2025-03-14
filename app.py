import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# Initialize session state for storing model predictions
if 'preds_rf' not in st.session_state:
    st.session_state.preds_rf = None
if 'preds_xgb' not in st.session_state:
    st.session_state.preds_xgb = None
if 'preds_cnn' not in st.session_state:
    st.session_state.preds_cnn = None

# Streamlit App
st.title("Machine Learning Model Comparison App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded:", df.head())

    # Split data
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Reshape for CNN
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train models
    if st.button("Train Random Forest"):
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features=0.7,
            bootstrap=False,
            min_samples_split=5,
            random_state=42
        )
        rf.fit(X_train, y_train)
        st.session_state.preds_rf = rf.predict(X_test)
        st.write("RF - MSE:", mean_squared_error(y_test, st.session_state.preds_rf))
        st.write("RF - R²:", r2_score(y_test, st.session_state.preds_rf))

    if st.button("Train XGBoost"):
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=9,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            gamma=0.25,
            min_child_weight=1
        )
        xgb.fit(X_train, y_train)
        st.session_state.preds_xgb = xgb.predict(X_test)
        st.write("XGB - MSE:", mean_squared_error(y_test, st.session_state.preds_xgb))
        st.write("XGB - R²:", r2_score(y_test, st.session_state.preds_xgb))

    if st.button("Train CNN"):
        cnn_model = Sequential([
            Conv1D(filters=256, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
            BatchNormalization(),  
            Conv1D(filters=128, kernel_size=1, activation='relu'),
            BatchNormalization(),  
            Conv1D(filters=64, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),  
            Dense(128, activation='relu'),  
            Dense(64, activation='relu'),
            Dense(1)  # Regression output layer
        ])

        optimizer = Adam(learning_rate=0.0005)
        cnn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        checkpoint = ModelCheckpoint('best_cnn_model.h5', save_best_only=True, monitor='val_loss')
        
        history = cnn_model.fit(X_train_cnn, y_train, epochs=150, batch_size=16, validation_data=(X_test_cnn, y_test), callbacks=[early_stopping, lr_scheduler], verbose=1)
        st.session_state.preds_cnn = cnn_model.predict(X_test_cnn)
        st.write("CNN - MSE:", mean_squared_error(y_test, st.session_state.preds_cnn))
        st.write("CNN - R²:", r2_score(y_test, st.session_state.preds_cnn))

        # Plot Loss Curve
        st.write("### CNN Training Loss Curve")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.legend()
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)

    if st.button("Scatter Plot Performance"):
        if st.session_state.preds_rf is not None and st.session_state.preds_xgb is not None and st.session_state.preds_cnn is not None:
            st.write("### Model Performance Visualization")
            fig, ax = plt.subplots()
            ax.scatter(y_test, st.session_state.preds_rf, label="Random Forest", alpha=0.5)
            ax.scatter(y_test, st.session_state.preds_xgb, label="XGBoost", alpha=0.5)
            ax.scatter(y_test, st.session_state.preds_cnn, label="CNN", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
        else:
            st.warning("Please train models first before generating plots.")

    if st.button("Line Chart Performance"):
        if st.session_state.preds_rf is not None and st.session_state.preds_xgb is not None and st.session_state.preds_cnn is not None:
            st.write("### Model Performance Comparison")
            fig_line, ax_line = plt.subplots()
            models = ["Random Forest", "XGBoost", "CNN"]
            mse_values = [mean_squared_error(y_test, st.session_state.preds_rf), mean_squared_error(y_test, st.session_state.preds_xgb), mean_squared_error(y_test, st.session_state.preds_cnn)]
            ax_line.plot(models, mse_values, marker='o', linestyle='-', label="MSE")
            ax_line.set_ylabel("MSE")
            ax_line.set_title("Model Performance Comparison")
            ax_line.legend()
            st.pyplot(fig_line)
        else:
            st.warning("Please train models first before generating plots.")
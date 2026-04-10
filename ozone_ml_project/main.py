from src.data.load_data import load_processed_data
from src.data.validate_data import validate_dataframe
from src.training.train_xgb import split_data, train_xgb, evaluate_model
import matplotlib.pyplot as plt
import pandas as pd
from src.training.train_xgb import (
    split_data,
    train_xgb,
    evaluate_model,
    time_series_cv
)
import shap
import numpy as np
from src.models.lstm_model import train_lstm, evaluate_lstm

def main():

    # Load data
    df = load_processed_data("data/processed/DATA.csv")
    validate_dataframe(df)

    print("\nData ready for modeling.")
    
    # -------------------------------
    # TIME FEATURES (Phase 5 carryover)
    # -------------------------------
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear

    # -------------------------------
    # PHASE 7 FEATURES (RESIDUAL-DRIVEN)
    # # -------------------------------
    # df.drop(columns=["O3_lag1"], inplace=True)
    # Volatility (captures spikes)
    df["O3_volatility"] = df["Ozone (µg/m³)"].rolling(3).std()

    # Change in NOx (captures sudden emissions)
    df["NOx_change"] = df["NOx (ppb)"].diff()


    # Rolling mean (short-term smoothing)
    df["O3_roll3"] = df["Ozone (µg/m³)"].shift(1).rolling(3).mean()

    # High pollution regime flag
    df["high_NOx"] = (df["NOx (ppb)"] > df["NOx (ppb)"].quantile(0.75)).astype(int)

    # Drop NaNs created by rolling/diff
    df = df.dropna()
    # Split
    X_train, X_test, y_train, y_test = split_data(df, "Ozone (µg/m³)")

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Train model
    model = train_xgb(X_train, y_train)

    print("\nModel trained.")

    # Evaluate
    y_pred, mae, r2 = evaluate_model(model, X_test, y_test)

    print("\n--- MODEL PERFORMANCE ---")
    print("MAE:", mae)
    print("R2:", r2)
    
    # -------------------------------
    # TIME SERIES CROSS VALIDATION
    # -------------------------------
    print("\nRunning Time Series Cross Validation...")

    cv_mae, cv_r2 = time_series_cv(df, "Ozone (µg/m³)")

    print("\n--- CROSS VALIDATION RESULTS ---")
    print(f"CV MAE: {cv_mae:.4f}")
    print(f"CV R2: {cv_r2:.4f}")
        
    plt.figure(figsize=(12,5))

    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")

    plt.title("Ozone Prediction (XGBoost)")
    plt.legend()
    plt.show()

    # Residuals
    # -------------------------------
    # RESIDUAL ANALYSIS (IMPROVED)
    # -------------------------------
    residuals = y_test - y_pred

    res_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Residual": residuals
    })

    print("\nTop Error Cases:")
    print(res_df.reindex(residuals.abs().sort_values(ascending=False).index).head(10))
        
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop Features:\n", importance.head(10))

    importance.head(10).plot(kind="barh")
    plt.title("Top Feature Importance")
    plt.show()

    # -------------------------------
    # SHAP ANALYSIS
    # -------------------------------
    print("\nRunning SHAP analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)
    shap.dependence_plot("Leighton_ratio", shap_values, X_test)

    print("\nRunning LSTM model...")

    lstm_model, scaler, X_test_lstm, y_test_lstm = train_lstm(df)

    preds_lstm, actual_lstm, mae_lstm, r2_lstm = evaluate_lstm(
        lstm_model, scaler, X_test_lstm, y_test_lstm
    )

    print("\n--- LSTM PERFORMANCE ---")
    print(f"LSTM MAE: {mae_lstm:.4f}")
    print(f"LSTM R2: {r2_lstm:.4f}")


    # -------------------------------
    # LSTM PREDICTION PLOT
    # -------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(actual_lstm, label="Actual")
    plt.plot(preds_lstm, label="LSTM Predicted")
    plt.title("LSTM Ozone Prediction")
    plt.legend()
    plt.grid()
    plt.show()


    # -------------------------------
    # LSTM RESIDUAL ANALYSIS
    # -------------------------------
    lstm_residuals = actual_lstm - preds_lstm

    plt.figure(figsize=(12, 4))
    plt.plot(lstm_residuals)
    plt.title("LSTM Residuals Over Time")
    plt.grid()
    plt.show()


    plt.figure(figsize=(6, 4))
    plt.hist(lstm_residuals, bins=30)
    plt.title("LSTM Residual Distribution")
    plt.grid()
    plt.show()


    # -------------------------------
    # TOP ERROR CASES (LSTM)
    # -------------------------------
    lstm_res_df = pd.DataFrame({
        "Actual": actual_lstm,
        "Predicted": preds_lstm,
        "Residual": lstm_residuals
    })

    print("\nTop LSTM Error Cases:")
    print(
        lstm_res_df.iloc[
            np.argsort(np.abs(lstm_residuals))[::-1]
        ].head(10)
    )

if __name__ == "__main__":
    main()
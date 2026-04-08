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
    # -------------------------------

    # Volatility (captures spikes)
    df["O3_volatility"] = df["Ozone (µg/m³)"].rolling(3).std()

    # Change in NOx (captures sudden emissions)
    df["NOx_change"] = df["NOx (ppb)"].diff()

    # Interaction term (nonlinear chemistry)
    df["NO2_O3_interaction"] = df["NO2 (µg/m³)"] * df["Ozone (µg/m³)"]

    # # Rolling mean (short-term smoothing)
    # df["O3_roll3"] = df["Ozone (µg/m³)"].shift(1).rolling(3).mean()

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
    shap.dependence_plot("NO2_O3_interaction", shap_values, X_test)
    shap.dependence_plot("Leighton_ratio", shap_values, X_test)

if __name__ == "__main__":
    main()
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
    residuals = y_test.values - y_pred

    plt.figure(figsize=(12,4))
    plt.plot(residuals)
    plt.title("Residuals (Errors)")
    plt.show()
    
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop Features:\n", importance.head(10))

    importance.head(10).plot(kind="barh")
    plt.title("Top Feature Importance")
    plt.show()


if __name__ == "__main__":
    main()
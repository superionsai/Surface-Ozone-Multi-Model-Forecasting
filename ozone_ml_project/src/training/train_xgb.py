from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import numpy as np


# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
def split_data(df, target_col):

    y = df[target_col]
    X = df.drop(columns=[target_col])

    split = int(0.8 * len(df))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    return X_train, X_test, y_train, y_test


# -------------------------------
# TUNED XGBOOST MODEL
# -------------------------------
def train_xgb(X_train, y_train):

    model = XGBRegressor(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# -------------------------------
# SINGLE SPLIT EVALUATION
# -------------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, mae, r2


# -------------------------------
# TIME SERIES CROSS VALIDATION
# -------------------------------
def time_series_cv(df, target_col):

    y = df[target_col]
    X = df.drop(columns=[target_col])

    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = train_xgb(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.4f}, R2: {r2:.4f}")

        mae_scores.append(mae)
        r2_scores.append(r2)

    return np.mean(mae_scores), np.mean(r2_scores)
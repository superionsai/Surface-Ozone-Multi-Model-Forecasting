# Phase Plan: Hourly Ozone Forecasting (Exhaustive Research Grade)

## Phase 1: Data Preprocessing (Completed)
- Cleaned 6 specific stations independently.
- Handled missing values (interpolated medium importance features, dropped worst-case missing NO2 rows).
- Output processed data to `data/processed/`.

## Phase 2: Advanced Feature Engineering & Time-Series Prep
*   **Step 2.1 - PM & Chemical Proxies:** Update `policy.py` to ensure `PM2.5` and `PM10` bypass any exclusion filters. Add logic to compute `VOC/NOx` ratios using available proxies (like Benzene/Toluene) to map the NOX-limited vs VOC-limited regimes.
*   **Step 2.2 - Cyclical Time Encodings:** Standard timestamps are ordinal and confuse models (e.g., Hour 23 jumps to Hour 0). We will write a function to create `hour_sin`, `hour_cos`, `month_sin`, and `month_cos` columns.
*   **Step 2.3 - Meteorological Vectors:** Convert Wind Speed and Wind Direction into `Wind_U` and `Wind_V` vector components.
*   **Step 2.4 - LSTM Sequence Generator:** Write a robust PyTorch `Dataset` class (`src/data/sequence_gen.py`) that uses a sliding window (e.g., T-24 to T-1) to create strictly isolated 3D tensors `[samples, timesteps, features]`.

## Phase 3: Rigid Model Architectures & Initial Training
*   **Step 3.1 - XGBoost Configuration:** Build `src/training/train_xgb.py`. Implement `TimeSeriesSplit` (n_splits=5) for cross-validation to prevent future-data leakage. Tune `max_depth`, `subsample`, and `colsample_bytree` to force the model to look at minor features (like PM and SO2).
*   **Step 3.2 - PyTorch LSTM Architecture:** In `lstm_model.py`, construct the exact network: `Input -> LSTM (hidden=64, layers=2, dropout=0.2) -> BatchNorm -> Fully Connected -> Output`.
*   **Step 3.3 - LSTM Training Loop:** Implement training with an Adam optimizer, `MSELoss`, and strict Early Stopping based on a sequential validation holdout (e.g., the last 15% of the timeline).

## Phase 4: Exhaustive Individual Evaluation (Research-Grade Reporting)
*Every single model (XGBoost, LSTM) will undergo this exhaustive analysis suite individually. Results will be saved in `reports/figures/phase4_individual/<model_name>/<station_name>/`.*

*   **Step 4.1 - Diurnal & Seasonal Dynamics:** 
    - Diurnal: Average 24-hour curve with std-dev bands.
    - Seasonal: Box plots comparing errors across Summer, Monsoon, Post-Monsoon, and Winter.
*   **Step 4.2 - High-Resolution & Distributional Analysis:**
    - Line graphs for three contiguous 7-day periods (peaks and dips).
    - Scatter & Q-Q plots to detect systematic underestimation/overestimation.
*   **Step 4.3 - AQI Categorical Performance:**
    - Generate **AQI Confusion Matrices** using National Ozone breakpoints.
    - Calculate Accuracy/F1-Score for predicting "Poor" or "Severe" air quality events.
*   **Step 4.4 - Advanced Scientific Statistics:**
    - Calculate **Index of Agreement (IA)**, **Percent Bias (PBIAS)**, and **Peak Prediction Accuracy (PPA)**.
    - Add **Normalized Mean Bias (NMB)**, **Normalized Mean Error (NME)**, and **FAC2** (Fraction of predictions within a factor of 2).
    - Generate an **Error-Feature Correlation Matrix** to identify where chemistry breaks the model.
*   **Step 4.5 - Distribution & Variance Analysis:**
    - **Taylor Diagrams:** A single plot showing the Correlation (R), RMSE, and Standard Deviation of the model relative to observations (Standard for Nature Scientific Reports).
    - **Probability Density Function (PDF) Matching:** Overlaying the actual and predicted distributions to ensure the model captures the "extremes" (fat tails) of the data.
*   **Step 4.6 - Automated Insight Engine:** 
    - Generate a consolidated `report_summary.txt` for each station.
    - Include automated "Graph Observations" (e.g., detecting phase lags, high-concentration biases, or seasonal drifts).
*   **Step 4.7 - SHAP Interpretability (XGBoost):** 
    - Summary and dependence plots to explain feature importance globally and locally.

## Phase 5: Technical Hybridization & Ensembling
*Ensembles will undergo the exact same Phase 4 plotting suite. All plots will be saved in `reports/figures/phase5_ensemble/<ensemble_type>/`.*
*   **Step 5.1 - Non-Linear Meta-Learner (Stacking):** Train a Random Forest using `[y_xgb, y_lstm, hour_sin, hour_cos, month]`.
*   **Step 5.2 - Sequential Residual Boosting:** Train LSTM purely to predict XGBoost residuals. Final outputs `XGB_Base + LSTM_Residual`.
*   **Step 5.3 - Feature Embedding:** Append LSTM hidden states into XGBoost tabular dataset.
*   **Step 5.4 - Ensemble Evaluation:** Run all Phase 4 plotting functions on the hybrid outputs.
    *   *Save Paths:* `reports/figures/phase5_ensemble/<ensemble_type>/<plot_category>/station_name_plot.png`
    *   *Save Paths:* `reports/metrics/phase5_ensemble_<ensemble_type>_stats.csv`

## Phase 6: Final Paper Extraction
*   **Step 6.1 - Consolidated Feature Interpretability:** Deep dive into SHAP interpretations for the final ensemble.
*   **Step 6.2 - Publishing Format:** Ensure all saved plots in `reports/figures/` are exported at high-DPI (300) PDF/PNG formats, organized perfectly for direct insertion into the research paper.

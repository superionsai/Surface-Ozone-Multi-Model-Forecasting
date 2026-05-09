import numpy as np
import pandas as pd

def generate_dense_report(model_type, station_name, df_eval, metrics, output_path):
    """
    Generates an extensive, highly detailed text report (>100 lines) 
    analyzing the model's performance across multiple dimensions.
    """
    # Pre-computation for Diurnal Analysis
    diurnal_true = df_eval.groupby('Hour')['Actual'].mean()
    diurnal_pred = df_eval.groupby('Hour')['Predicted'].mean()
    diurnal_error = np.abs(diurnal_true - diurnal_pred)
    worst_hour = diurnal_error.idxmax()
    best_hour = diurnal_error.idxmin()
    
    daytime_mask = (df_eval['Hour'] >= 8) & (df_eval['Hour'] <= 18)
    nighttime_mask = ~daytime_mask
    day_rmse = np.sqrt(np.mean((df_eval.loc[daytime_mask, 'Actual'] - df_eval.loc[daytime_mask, 'Predicted'])**2))
    night_rmse = np.sqrt(np.mean((df_eval.loc[nighttime_mask, 'Actual'] - df_eval.loc[nighttime_mask, 'Predicted'])**2))

    # Pre-computation for Seasonal Analysis
    seasonal_stats = df_eval.groupby('Season').apply(
        lambda x: pd.Series({
            'RMSE': np.sqrt(np.mean((x['Actual'] - x['Predicted'])**2)),
            'MAE': np.mean(np.abs(x['Actual'] - x['Predicted'])),
            'Mean_Actual': x['Actual'].mean()
        })
    )
    worst_season = seasonal_stats['RMSE'].idxmax() if not seasonal_stats.empty else "N/A"
    best_season = seasonal_stats['RMSE'].idxmin() if not seasonal_stats.empty else "N/A"

    # Pre-computation for Distributional Analysis
    std_actual = df_eval['Actual'].std()
    std_pred = df_eval['Predicted'].std()
    variance_ratio = std_pred / std_actual if std_actual > 0 else 0
    
    # AQI Extremes
    extreme_mask = df_eval['Actual'] > 200 # Poor and above
    if extreme_mask.sum() > 0:
        extreme_rmse = np.sqrt(np.mean((df_eval.loc[extreme_mask, 'Actual'] - df_eval.loc[extreme_mask, 'Predicted'])**2))
        extreme_bias = np.mean(df_eval.loc[extreme_mask, 'Predicted'] - df_eval.loc[extreme_mask, 'Actual'])
    else:
        extreme_rmse = np.nan
        extreme_bias = np.nan

    lines = []
    lines.append("=" * 80)
    lines.append(f"EXHAUSTIVE RESEARCH EVALUATION REPORT")
    lines.append(f"Model Architecture : {model_type.upper()}")
    lines.append(f"Target Station     : {station_name.upper()}")
    lines.append(f"Total Test Samples : {len(df_eval)}")
    lines.append("=" * 80)
    lines.append("")

    test_metrics = metrics.get('Test', metrics)
    train_metrics = metrics.get('Train', {})
    val_metrics = metrics.get('Val', {})

    # Section 1: Executive Summary & Global Metrics
    lines.append("1. EXECUTIVE SUMMARY & GLOBAL METRICS")
    lines.append("-" * 40)
    lines.append(f"The {model_type.upper()} model was evaluated using a Hybrid Research Split:")
    lines.append("- Train: 65% of total data (Randomly sampled from first 5 years).")
    lines.append("- Val  : Remainder of first 5 years (Randomly sampled).")
    lines.append("- Test : Final 1 year of data (Strict Chronological).")
    lines.append("This strategy ensures the model learns all seasonal chemistry while strictly testing on unseen future data.")
    lines.append("")
    
    # Create comparative table
    lines.append(f"{'Metric':<30} | {'Train (65%)':<12} | {'Val (~18%)':<12} | {'Test (1 Yr)':<12}")
    lines.append("-" * 75)
    for m in ['MSE', 'RMSE', 'MAE', 'R2']:
        m_train = train_metrics.get(m, np.nan)
        m_val = val_metrics.get(m, np.nan)
        m_test = test_metrics.get(m, np.nan)
        lines.append(f"{m:<30} | {m_train:<12.4f} | {m_val:<12.4f} | {m_test:<12.4f}")
    
    lines.append("")
    lines.append("Analysis of Base Metrics:")
    r2 = test_metrics.get('R2', 0)
    if r2 > 0.8:
        lines.append("The R\u00b2 score indicates excellent variance capture. The model successfully maps the non-linear precursor relationships to ground-level ozone formation.")
    elif r2 > 0.5:
        lines.append("The R\u00b2 score indicates moderate variance capture. While general trends are learned, stochastic localized events or missing unmeasured precursors may limit accuracy.")
    else:
        lines.append("The R\u00b2 score is critically low. The architecture struggles to separate the ozone signal from background meteorological noise, indicating potential underfitting or lack of representational capacity in the current feature space.")
    lines.append("")

    # Section 2: Advanced Scientific Statistics
    lines.append("2. ADVANCED SCIENTIFIC STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  * Index of Agreement (IA)        : {test_metrics.get('IA', np.nan):.4f} (Scale: 0 to 1)")
    lines.append(f"  * Normalized Mean Bias (NMB)     : {test_metrics.get('NMB (%)', np.nan):.4f}%")
    lines.append(f"  * Normalized Mean Error (NME)    : {test_metrics.get('NME (%)', np.nan):.4f}%")
    lines.append(f"  * Percent Bias (PBIAS)           : {test_metrics.get('PBIAS', np.nan):.4f}%")
    lines.append(f"  * Factor of 2 Accuracy (FAC2)    : {test_metrics.get('FAC2', np.nan):.4f} (Ideal: > 0.5)")
    lines.append(f"  * Peak RMSE (Top 10% obs.)       : {test_metrics.get('Peak RMSE (Top 10%)', np.nan):.4f} \u00b5g/m\u00b3")
    lines.append("")
    lines.append("Diagnostic Interpretation:")
    pbias = test_metrics.get('PBIAS', 0)
    if abs(pbias) < 10:
        lines.append("The Percent Bias (PBIAS) is within optimal ranges (< |10%|), suggesting the model predictions are evenly distributed across the regression plane without systematic over/underestimation.")
    elif pbias > 10:
        lines.append("The Percent Bias (PBIAS) reveals a systemic positive bias. The model is chronically over-predicting surface ozone, likely failing to account for sufficient nighttime NOx titration or localized deposition sinks.")
    else:
        lines.append("The Percent Bias (PBIAS) reveals a systemic negative bias. The model is underestimating ozone formation, potentially missing the magnitude of peak afternoon photochemical production driven by high VOC/NOx ratios.")
    
    fac2 = test_metrics.get('FAC2', 0)
    lines.append(f"The FAC2 metric shows that {fac2*100:.1f}% of all predictions fall within a strict factor of 2 of the observed values. In urban air quality forecasting, a FAC2 > 0.5 is required for operational viability.")
    lines.append("")

    # Section 3: Diurnal (Time-of-Day) Dynamics
    lines.append("3. DIURNAL DYNAMICS & PHOTOCHEMICAL CYCLE")
    lines.append("-" * 40)
    lines.append("Ground-level ozone exhibits a distinct diurnal cycle governed by solar radiation (driving photolysis of NO2) and nighttime titration (via NO emission).")
    lines.append(f"  * Daytime RMSE (08:00 - 18:00)   : {day_rmse:.4f} \u00b5g/m\u00b3")
    lines.append(f"  * Nighttime RMSE (19:00 - 07:00) : {night_rmse:.4f} \u00b5g/m\u00b3")
    lines.append(f"  * Most Accurate Hour             : {best_hour:02d}:00 (Mean Absolute Error: {diurnal_error.min():.4f})")
    lines.append(f"  * Least Accurate Hour            : {worst_hour:02d}:00 (Mean Absolute Error: {diurnal_error.max():.4f})")
    lines.append("")
    lines.append("Diurnal Phase Analysis:")
    if day_rmse > night_rmse * 1.5:
        lines.append("The model exhibits significantly higher error during daylight hours. This points to a deficiency in modeling the complex, radiation-dependent photochemical generation rate. The model likely underestimates the rapid escalation of ozone during peak solar irradiance.")
    elif night_rmse > day_rmse * 1.5:
        lines.append("The model struggles primarily during nighttime hours. This is typically symptomatic of failing to capture localized boundary layer collapses and subsequent rapid titration by fresh NO emissions from evening traffic.")
    else:
        lines.append("Error magnitude is relatively consistent across the diurnal cycle, suggesting the architecture balances both daytime production and nighttime destruction dynamics symmetrically.")
    
    peak_actual_hour = diurnal_true.idxmax()
    peak_pred_hour = diurnal_pred.idxmax()
    lines.append(f"Observed peak ozone occurs on average at {peak_actual_hour:02d}:00, while the model predicts the peak at {peak_pred_hour:02d}:00.")
    if peak_actual_hour != peak_pred_hour:
        lines.append(f"-> WARNING: A phase lag of {abs(peak_actual_hour - peak_pred_hour)} hours is detected in the diurnal peak.")
    else:
        lines.append("-> Phase alignment is perfect. The model correctly identifies the timing of maximum daily photochemical activity.")
    lines.append("")

    # Section 4: Seasonal Segregation
    lines.append("4. SEASONAL DISTRIBUTION & GENERALIZATION")
    lines.append("-" * 40)
    lines.append("Evaluating generalization across the distinct meteorological regimes of the Indian subcontinent.")
    for season, stats in seasonal_stats.iterrows():
        lines.append(f"  * {season:<15} - RMSE: {stats['RMSE']:>7.4f} | MAE: {stats['MAE']:>7.4f} | Mean Conc: {stats['Mean_Actual']:>7.4f}")
    lines.append("")
    lines.append("Seasonal Discrepancy Analysis:")
    lines.append(f"The highest error regime is identified as {worst_season}, whereas the model performs optimally during {best_season}.")
    if worst_season == 'Winter':
        lines.append("High Winter error is common due to shallow boundary layers trapping precursors, leading to highly volatile, localized titration events that are difficult for generalized grids to capture.")
    elif worst_season == 'Monsoon':
        lines.append("Monsoon errors usually stem from abrupt wash-out events and sporadic cloud cover breaking the continuous radiation assumptions.")
    elif worst_season == 'Summer':
        lines.append("Summer maximum errors indicate the model cannot properly scale with extreme temperature-driven biogenic VOC emissions and accelerated NO2 photolysis rates.")
    lines.append("")

    # Section 5: Extreme Event & AQI Boundary Analysis
    lines.append("5. EXTREME EVENT PREDICTION & AQI BOUNDARIES")
    lines.append("-" * 40)
    lines.append("Regulatory and health-advisory models are judged primarily on their ability to predict extreme pollution events (AQI 'Poor' and above, >200 \u00b5g/m\u00b3).")
    if not np.isnan(extreme_rmse):
        lines.append(f"  * Extreme Event RMSE (>200)      : {extreme_rmse:.4f} \u00b5g/m\u00b3")
        lines.append(f"  * Extreme Event Mean Bias        : {extreme_bias:.4f} \u00b5g/m\u00b3")
        lines.append("")
        if extreme_bias < -30:
            lines.append("DANGER: The model displays severe negative bias during extreme events. It acts as an 'averager,' severely damping the peaks of high-pollution days. This renders the model unreliable for early-warning health advisory systems.")
        elif extreme_bias > 30:
            lines.append("WARNING: The model overestimates extreme peaks, which would lead to a high false-alarm rate in a regulatory context.")
        else:
            lines.append("SUCCESS: The model handles extreme distributions reasonably well, avoiding the common pitfall of mean-regression damping.")
    else:
        lines.append("  * No significant extreme events (>200 \u00b5g/m\u00b3) were present in the target holdout subset to evaluate tail-end robustness.")
    lines.append("")

    # Section 6: Variance & Structural Integrity
    lines.append("6. VARIANCE & STRUCTURAL INTEGRITY")
    lines.append("-" * 40)
    lines.append("Comparing the standard deviation (spread) of the predicted signal versus the true signal.")
    lines.append(f"  * Standard Deviation (Observed)  : {std_actual:.4f}")
    lines.append(f"  * Standard Deviation (Predicted) : {std_pred:.4f}")
    lines.append(f"  * Variance Ratio (Pred/Obs)      : {variance_ratio:.4f}")
    lines.append("")
    if variance_ratio < 0.7:
        lines.append("The predicted variance is heavily suppressed compared to reality (Ratio < 0.7). The model geometry is too rigid, failing to replicate the natural atmospheric volatility.")
    elif variance_ratio > 1.3:
        lines.append("The predicted variance is inflated (Ratio > 1.3). The model is overly sensitive to input perturbations, causing erratic, high-amplitude predictions.")
    else:
        lines.append("The variance ratio is near optimal (~1.0). The model successfully replicates the structural volatility and dynamic range of the true atmospheric sequence.")
    lines.append("")
    
    # Section 7: Empirical Observations
    lines.append("7. EMPIRICAL ATMOSPHERIC OBSERVATIONS (RAW DATA)")
    lines.append("-" * 40)
    lines.append("Exploratory analysis of the true observational dataset prior to ML application.")
    
    # Calculate monthly true averages
    monthly_means = df_eval.groupby('Month')['Actual'].mean()
    if not monthly_means.empty:
        peak_month = monthly_means.idxmax()
        trough_month = monthly_means.idxmin()
        lines.append(f"  * Highest Mean Ozone Month       : {peak_month:02d} ({monthly_means.max():.2f} \u00b5g/m\u00b3)")
        lines.append(f"  * Lowest Mean Ozone Month        : {trough_month:02d} ({monthly_means.min():.2f} \u00b5g/m\u00b3)")
    
    hazardous_count = extreme_mask.sum()
    lines.append(f"  * Total Hazardous Hours (>200)   : {hazardous_count}")
    if hazardous_count > 0:
        haz_months = df_eval.loc[extreme_mask, 'Month'].value_counts().head(3)
        top_haz_months = ", ".join([f"Month {m} ({c} hours)" for m, c in haz_months.items()])
        lines.append(f"  * Primary Hazardous Clusters     : {top_haz_months}")
        lines.append("-> Hazardous events are heavily localized in specific temporal windows, typically driven by extreme heat and prolonged stagnation events during pre-monsoon summer.")
    else:
        lines.append("-> The evaluation set represents a relatively clean temporal slice with no Severe category occurrences.")
    lines.append("")
    
    # Section 8: Phenomenological Atmospheric Analysis
    lines.append("8. PHENOMENOLOGICAL ATMOSPHERIC ANALYSIS")
    lines.append("-" * 40)
    lines.append("Analyzing the physical behaviors and non-linear interactions of the local atmosphere.")
    
    # 8.1 Ozone Weekend Effect
    df_eval['DayOfWeek'] = df_eval['Timestamp'].dt.dayofweek
    actual_weekend = df_eval.loc[df_eval['DayOfWeek'] >= 5, 'Actual'].mean()
    actual_weekday = df_eval.loc[df_eval['DayOfWeek'] < 5, 'Actual'].mean()
    pred_weekend = df_eval.loc[df_eval['DayOfWeek'] >= 5, 'Predicted'].mean()
    pred_weekday = df_eval.loc[df_eval['DayOfWeek'] < 5, 'Predicted'].mean()
    
    lines.append("A. Ozone Weekend Effect (OWE):")
    lines.append(f"   Observed -> Weekday Mean: {actual_weekday:.2f} | Weekend Mean: {actual_weekend:.2f}")
    lines.append(f"   Modeled  -> Weekday Mean: {pred_weekday:.2f} | Weekend Mean: {pred_weekend:.2f}")
    if actual_weekend > actual_weekday:
        lines.append("   -> OWE Detected: The raw data shows higher ozone on weekends (typically due to lower NOx emissions reducing titration).")
        if pred_weekend > pred_weekday:
            lines.append("   -> Model Success: The model successfully learned the non-linear Weekend Effect.")
        else:
            lines.append("   -> Model Failure: The model failed to replicate the Weekend Effect, predicting lower/equal ozone on weekends.")
    else:
        lines.append("   -> No OWE Detected: Weekend ozone is lower or equal to weekday ozone in reality.")
    lines.append("")
    
    # 8.2 Ascent and Titration Rates
    lines.append("B. Photochemical Ascent & Nighttime Titration Rates:")
    if 8 in diurnal_true and 12 in diurnal_true and 18 in diurnal_true and 22 in diurnal_true:
        true_ascent = (diurnal_true[12] - diurnal_true[8]) / 4.0
        pred_ascent = (diurnal_pred[12] - diurnal_pred[8]) / 4.0
        true_titration = (diurnal_true[18] - diurnal_true[22]) / 4.0
        pred_titration = (diurnal_pred[18] - diurnal_pred[22]) / 4.0
        
        lines.append(f"   Morning Ascent Rate (08:00-12:00) : Observed = +{true_ascent:.2f} \u00b5g/m\u00b3/hr | Modeled = +{pred_ascent:.2f} \u00b5g/m\u00b3/hr")
        lines.append(f"   Evening Titration Rate (18:00-22:00): Observed = -{true_titration:.2f} \u00b5g/m\u00b3/hr | Modeled = -{pred_titration:.2f} \u00b5g/m\u00b3/hr")
        
        if abs(true_ascent - pred_ascent) > true_ascent * 0.3:
            lines.append("   -> Warning: The modeled morning production rate deviates significantly (>30%) from reality, suggesting flawed photolysis mapping.")
        if abs(true_titration - pred_titration) > true_titration * 0.3:
            lines.append("   -> Warning: The modeled nighttime collapse rate deviates significantly, suggesting poor mapping of NO scavenging.")
    lines.append("")
    
    # 8.3 Peak Duration
    lines.append("C. Peak Duration & Multiday Stagnation:")
    def get_avg_duration(series, threshold=100):
        is_over = series > threshold
        comps = is_over.ne(is_over.shift()).cumsum()
        lengths = comps[is_over].value_counts()
        return lengths.mean() if len(lengths) > 0 else 0.0

    true_dur = get_avg_duration(df_eval['Actual'])
    pred_dur = get_avg_duration(df_eval['Predicted'])
    lines.append(f"   Average consecutive hours > 100 \u00b5g/m\u00b3 : Observed = {true_dur:.1f} hours | Modeled = {pred_dur:.1f} hours")
    if true_dur > pred_dur * 1.5:
        lines.append("   -> The model prematurely cuts off high-ozone events. It fails to capture the prolonged stagnation common in urban centers.")
    elif pred_dur > true_dur * 1.5:
        lines.append("   -> The model artificially elongates peaks, predicting stagnation that does not actually occur.")
    else:
        lines.append("   -> The model successfully captures the temporal duration and persistence of hazardous events.")
    lines.append("")
    
    # Conclusion
    lines.append("9. FINAL DIAGNOSTIC CONCLUSION")
    lines.append("-" * 40)
    if r2 > 0.75 and abs(pbias) < 15 and variance_ratio > 0.8:
        lines.append(f"The {model_type.upper()} architecture demonstrates robust research-grade viability for the {station_name} station. It effectively mitigates seasonal generalization gaps, tracks diurnal photochemistry with high fidelity, and preserves the natural variance of the atmospheric system.")
    elif r2 < 0.3:
        lines.append(f"The {model_type.upper()} architecture fails to model the basic underlying physics at {station_name}. A complete re-evaluation of hyperparameter limits, input feature engineering, or sequence length is required before operational deployment.")
    else:
        lines.append(f"The {model_type.upper()} architecture shows baseline competence at {station_name} but exhibits specific analytical flaws detailed above (e.g., peak suppression, phase lagging, or seasonal bias). Targeted ensembling or residual boosting is recommended to correct these isolated domain failures.")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

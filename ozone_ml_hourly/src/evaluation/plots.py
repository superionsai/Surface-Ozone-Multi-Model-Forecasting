import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set research-grade plotting style
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_diurnal_cycle(df_eval, output_path):
    """Plots the average 24-hour cycle with std deviation."""
    plt.figure(figsize=(10, 6))
    
    # Group by hour
    diurnal_true = df_eval.groupby('Hour')['Actual'].agg(['mean', 'std'])
    diurnal_pred = df_eval.groupby('Hour')['Predicted'].agg(['mean', 'std'])
    
    hours = diurnal_true.index
    
    plt.plot(hours, diurnal_true['mean'], label='Observed', color='black', linewidth=2)
    plt.fill_between(hours, diurnal_true['mean'] - diurnal_true['std'], diurnal_true['mean'] + diurnal_true['std'], color='gray', alpha=0.2)
    
    plt.plot(hours, diurnal_pred['mean'], label='Predicted', color='blue', linestyle='--', linewidth=2)
    plt.fill_between(hours, diurnal_pred['mean'] - diurnal_pred['std'], diurnal_pred['mean'] + diurnal_pred['std'], color='blue', alpha=0.1)
    
    plt.title('Mean Diurnal Cycle of Surface Ozone')
    plt.xlabel('Hour of Day')
    plt.ylabel('Ozone Concentration (\u00b5g/m\u00b3)')
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_seasonal_boxplots(df_eval, output_path):
    """Plots Boxplots of Absolute Error by Season."""
    df_eval['Absolute_Error'] = np.abs(df_eval['Actual'] - df_eval['Predicted'])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Season', y='Absolute_Error', data=df_eval, hue='Season', palette='Set2', legend=False)
    plt.title('Distribution of Absolute Error by Season')
    plt.ylabel('Absolute Error (\u00b5g/m\u00b3)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_high_res_dynamics(df_eval, output_path, days=7):
    """Plots a zoomed-in contiguous week to show rapid dynamics."""
    plt.figure(figsize=(14, 5))
    
    # Grab the first 'days' worth of data (assuming hourly data)
    subset = df_eval.head(days * 24).reset_index(drop=True)
    
    plt.plot(subset.index, subset['Actual'], label='Observed', color='black', alpha=0.8)
    plt.plot(subset.index, subset['Predicted'], label='Predicted', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'High-Resolution Dynamics ({days}-Day Snapshot)')
    plt.xlabel('Hours')
    plt.ylabel('Ozone Concentration (\u00b5g/m\u00b3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_pdf_overlay(df_eval, output_path):
    """Plots the Probability Density Function (PDF) of Actual vs Predicted."""
    plt.figure(figsize=(8, 6))
    
    sns.kdeplot(df_eval['Actual'], label='Observed', color='black', fill=True, alpha=0.1)
    sns.kdeplot(df_eval['Predicted'], label='Predicted', color='blue', fill=True, alpha=0.1)
    
    plt.title('Probability Density Function (PDF) of Ozone')
    plt.xlabel('Ozone Concentration (\u00b5g/m\u00b3)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_aqi_confusion(df_eval, output_path):
    """Plots a Confusion Matrix based on Indian AQI Breakpoints for Ozone (8-hr proxy)."""
    # Breakpoints (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
    bins = [-1, 50, 100, 200, 300, 400, 1000]
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    
    actual_binned = pd.cut(df_eval['Actual'], bins=bins, labels=labels)
    pred_binned = pd.cut(df_eval['Predicted'], bins=bins, labels=labels)
    
    cm = pd.crosstab(actual_binned, pred_binned, dropna=False)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('AQI Category Confusion Matrix')
    plt.xlabel('Predicted Category')
    plt.ylabel('Observed Category')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_taylor_diagram(df_eval, output_path):
    """Attempts to plot a Taylor Diagram using SkillMetrics."""
    try:
        import skill_metrics as sm
    except ImportError:
        print("SkillMetrics not installed. Skipping Taylor Diagram. (pip install SkillMetrics)")
        return
        
    plt.figure(figsize=(8, 8))
    
    # Calculate statistics for Taylor Diagram
    # sm.taylor_statistics(predicted, reference)
    taylor_stats = sm.taylor_statistics(df_eval['Predicted'].values, df_eval['Actual'].values)
    
    # Store standard deviations, correlation, and centered RMS difference
    sdev = np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]])
    crmsd = np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]])
    ccoef = np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]])
    
    # Generate the diagram
    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=['Observation', 'Model'],
                      markercolor='r', markerSize=10, 
                      tickRMS=[0.0, 20.0, 40.0, 60.0, 80.0],
                      tickRMSangle=115.0, colRMS='m', styleRMS=':',
                      titleRMS='on', titleRMSDangle=40.0,
                      colSTD='b', styleSTD='-.', titleSTD='on')
                      
    plt.title('Taylor Diagram')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_regression_scatter(df_eval, output_path):
    """Plots a Scatter plot of Model vs Measurements with 1:1 line and FAC2 boundaries."""
    from scipy.stats import linregress
    from sklearn.metrics import r2_score
    
    plt.figure(figsize=(8, 8))
    
    actual = df_eval['Actual'].values
    predicted = df_eval['Predicted'].values
    
    # Scatter plot
    plt.scatter(actual, predicted, color='red', s=10, alpha=0.6, label='Predictions')
    
    # 1:1 Line
    max_val = max(np.nanmax(actual), np.nanmax(predicted))
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='-', linewidth=2, label='1:1 Line')
    
    # FAC2 Lines (1:2 and 2:1)
    plt.plot([0, max_val], [0, max_val/2], color='black', linestyle='--', linewidth=1.5, label='FAC2 Boundary (1:2)')
    plt.plot([0, max_val/2], [0, max_val], color='black', linestyle='--', linewidth=1.5, label='FAC2 Boundary (2:1)')
    
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(actual, predicted)
    plt.plot(actual, intercept + slope * actual, color='blue', linestyle='-.', label='Best Fit')
    
    r2 = r2_score(actual, predicted)
    
    plt.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
    plt.title('Regression Scatter: Model vs Measurements')
    plt.xlabel('Measured Ozone (\u00b5g/m\u00b3)')
    plt.ylabel('Modeled Ozone (\u00b5g/m\u00b3)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_error_bar_chart(df_eval, output_path):
    """Plots a dual-axis bar chart showing R2 and RMSE across seasons."""
    from sklearn.metrics import r2_score, mean_squared_error
    
    seasons = df_eval['Season'].unique()
    r2_list = []
    rmse_list = []
    
    for s in seasons:
        subset = df_eval[df_eval['Season'] == s]
        if len(subset) > 1:
            r2_list.append(r2_score(subset['Actual'], subset['Predicted']))
            rmse_list.append(np.sqrt(mean_squared_error(subset['Actual'], subset['Predicted'])))
        else:
            r2_list.append(0)
            rmse_list.append(0)
            
    x = np.arange(len(seasons))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    rects1 = ax1.bar(x - width/2, r2_list, width, label='$R^2$', color='red', edgecolor='black', hatch='xx')
    ax1.set_ylabel('$R^2$', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, rmse_list, width, label='RMSE', color='magenta', edgecolor='black', hatch='//')
    ax2.set_ylabel('RMSE (\u00b5g/m\u00b3)', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(seasons)
    
    fig.tight_layout()
    plt.title('Performance Metrics by Season')
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_monthly_trends(df_eval, output_path):
    """Plots the mean actual vs predicted ozone per month."""
    monthly_stats = df_eval.groupby('Month').agg({'Actual': 'mean', 'Predicted': 'mean'}).reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_stats['Month'], monthly_stats['Actual'], marker='o', label='Actual Mean', color='black', linewidth=2)
    plt.plot(monthly_stats['Month'], monthly_stats['Predicted'], marker='s', label='Predicted Mean', color='blue', linestyle='--', linewidth=2)
    
    plt.title('Monthly Average Ozone Trends')
    plt.xlabel('Month')
    plt.ylabel('Ozone Concentration (\u00b5g/m\u00b3)')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hazardous_events(df_eval, output_path):
    """Plots isolated hazardous events (>200 \u00b5g/m\u00b3)."""
    hazardous = df_eval[df_eval['Actual'] > 200]
    
    plt.figure(figsize=(12, 5))
    if len(hazardous) > 0:
        plt.scatter(hazardous['Timestamp'], hazardous['Actual'], color='red', label='Actual > 200', alpha=0.8)
        plt.scatter(hazardous['Timestamp'], hazardous['Predicted'], color='blue', marker='x', label='Predicted', alpha=0.8)
        
        # Connect the actual and predicted with a vertical line
        for i, row in hazardous.iterrows():
            plt.plot([row['Timestamp'], row['Timestamp']], [row['Actual'], row['Predicted']], color='gray', linestyle=':', alpha=0.5)
            
    plt.axhline(y=200, color='red', linestyle='--', label='Hazardous Threshold (200)')
    plt.title(f'Hazardous Event Evaluation (N={len(hazardous)})')
    plt.xlabel('Date')
    plt.ylabel('Ozone Concentration (\u00b5g/m\u00b3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_shap_summary(model, X_scaled_df, output_path):
    """Generates SHAP beeswarm summary plot for XGBoost."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping. (pip install shap)")
        return
        
    # SHAP can be slow, so we take a sample if the dataset is large
    if len(X_scaled_df) > 2000:
        X_sample = shap.utils.sample(X_scaled_df, 2000)
    else:
        X_sample = X_scaled_df
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    """Plots native XGBoost feature importance (gain)."""
    importance_vals = model.feature_importances_
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance_vals})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(15) # Top 15
    
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='skyblue', edgecolor='black')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

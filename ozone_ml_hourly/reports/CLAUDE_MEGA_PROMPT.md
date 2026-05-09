# MEGA-PROMPT FOR COMPREHENSIVE RESEARCH PAPER GENERATION

**INSTRUCTIONS FOR THE USER:** 
Upload this file, along with your two reference PDFs and your station `report_summary.txt` files, to your LLM of choice. Then use the prompt block below.

---

## 🎯 THE PROMPT (Copy and paste this to the AI)

"Act as a Post-Doctoral Atmospheric Chemist and Senior Machine Learning Researcher. I am writing a high-impact journal paper on forecasting hourly surface ozone ($O_3$) in the highly polluted Delhi NCR region using Advanced Deep Learning (BiLSTM with Temporal Attention) and Tree-Based Ensembles (XGBoost).

I have attached a context document detailing my entire methodology, data engineering pipeline, model architectures, and station-specific results. I have also attached two reference papers from *Nature Scientific Reports* (s41598-021-01824-z and s41598-022-09619-6). 

Your task is to write a **massive, exhaustive, and publication-ready research paper**. Do not summarize. Expand on every concept. Use academic rigor. 

**I want you to write this section by section. Please start by writing just 'Section 1: Introduction and Literature Review' and 'Section 2: Study Area and Reference Paper Critique'. Wait for my approval before moving to the next sections.**"

---

## 🧠 CONTEXT PAYLOAD (The AI will read this)

### 1. The Reference Papers (To Extend or Challenge)
*   **Paper 1 (Doon Valley - 2021):** Used ML to simulate urban ozone variability. Achieved $R^2 \sim 0.7$. Emphasized that adding precursors ($CO$, $NO_x$) to meteorology is crucial.
*   **Paper 2 (Munich - 2022):** Proved that in-situ precursor data improves ML model variance capture by 15% ($R^2 = 0.87$). Showed model generalizability across cities.
*   **Our Critique & Extension:** Both papers use relatively clean, stationary data. They fail to address **Sensor Drift**, non-stationary baselines, and extreme winter titration in mega-cities like Delhi. We challenge the over-reliance on standard $R^2$ as an air quality metric. If a sensor degrades over 5 years (baseline drops), $R^2$ collapses even if the diurnal photochemical cycle is predicted perfectly. We solve this via Additive Decomposition.

### 2. Exhaustive Data Engineering Pipeline
Explain each step with physical and mathematical justification:
*   **Data Source:** 6 years of hourly data across 6 diverse Delhi stations (Anand Vihar, Bawana, Narela, North Campus, Punjabi Bagh, RK Puram).
*   **Diurnal Mean Filling:** Instead of linear interpolation which fails for 12+ hour gaps, we fill missing $NO_2$ and $O_3$ using the historical mean of that *specific hour*. Reason: Preserves the diurnal "heartbeat" required for continuous LSTM sequences without introducing artificial flatlines.
*   **Feature Engineering:**
    *   *Chemical Ratios*: Added $NO_2/NO$ proxies.
    *   *Cyclical Time*: Sin/Cos transforms of Hour, Day, Month to capture the Earth's rotation and orbit seamlessly.
*   **Winsorization:** Clipped the top 5% of target ozone values. Reason: Delhi sensors frequently malfunction, recording unphysical spikes (>300 µg/m³) due to localized combustion, not regional photochemistry.
*   **The "Anand Vihar Fix" (Additive Decomposition):** To combat severe sensor drift (baseline dropping by 70% in 2024), we implemented a strictly backward-looking 30-day (720-hour) rolling mean baseline. The model predicts the **Ozone_Anomaly** (Target = Raw - Baseline). This forces the model to learn the daily photochemical "pulse" rather than memorizing fluctuating absolute concentrations.
*   **Hybrid Research Split:** 
    *   *Train/Val*: 65% / 15% randomly sampled from the *first 5 years*. (Exposes the model to every type of seasonal anomaly).
    *   *Test*: Strict chronological holdout of the *last 1 year*. (Proves true forecasting ability on unseen future data).

### 3. Model Architectures & Justifications
*   **XGBoost Baseline:**
    *   *Loss Function:* Pseudo-Huber Error. Reason: More robust to outliers than MSE, crucial for chaotic urban air quality.
    *   *Monotonic Constraints:* Forced the model to respect basic physical laws (e.g., increasing solar radiation must not decrease ozone).
*   **Advanced BiLSTM (The Primary Innovation):**
    *   *Bidirectional:* Reads the sequence forwards and backwards, understanding how an accumulation phase leads to a peak.
    *   *Temporal Attention Layer:* Allows the network to "look back" at the previous 24 hours and assign dynamic weights. E.g., it learns that yesterday's 2 PM peak is highly correlated with today's 2 PM peak.
    *   *Residual Skip Connections:* Prevents vanishing gradients, allowing the model to retain long-term seasonal context while processing immediate hourly changes.

### 4. Station-by-Station Analytics & Quality
Explain that station performance varies wildly due to local topography and emission sources:
*   **Bawana & RK Puram (The Successes):** High $R^2$ (~0.74). Perfect diurnal phase alignment. The models successfully capture the daytime photochemical ascent and the nighttime NOx titration.
*   **Anand Vihar (The Anomaly):** Extremely low $R^2$. However, Diurnal Phase Analysis shows the model captures the exact *timing* of the peaks perfectly. The failure is in scale (due to sensor drift), not physics.
*   **Punjabi Bagh & Narela:** Moderate performance. Models exhibit a 1-2 hour "Phase Lag," predicting peaks slightly later than observed.
*   **The Weekend Effect (OWE):** The models successfully learned that Ozone is *higher* on weekends in Delhi, despite lower traffic. Reason: Lower $NO$ emissions on weekends means less nighttime titration (scavenging) of $O_3$.

### 5. The Argument Against R² (Crucial Section)
Argue fiercely against traditional ML metrics for Air Quality.
*   $R^2$ penalizes uniform baseline shifts heavily.
*   We propose that **Index of Agreement (IA)** and **Factor of 2 (FAC2)** are superior metrics for regulatory compliance.
*   **Diurnal Replication:** A model is only physically valid if it reproduces the correct hourly curve, regardless of global RMSE.

### 6. Visual Component Integration Guide
When writing, explicitly instruct where the following figures should be placed in the manuscript and what they represent:
*   `taylor_diagram.png`: Place in Model Comparison. Shows how XGB and LSTM differ in standard deviation and correlation compared to the observed reference point.
*   `diurnal_cycle.png`: Place in Phenomenological Analysis. Shows the model's ability to replicate the 2 PM peak and nighttime trough.
*   `feature_importance.png` / `shap_summary.png`: Place in Precursor Analysis. Validates the Munich paper by showing $NO_2$ and Solar Radiation dominate the tree splits.
*   `seasonal_boxplots.png`: Place in Seasonal Dynamics. Shows that Summer has the highest variance and highest error due to unmeasured biogenic VOCs.
*   `regression_scatter.png`: Place in Global Metrics.

---
**END OF CONTEXT PAYLOAD**

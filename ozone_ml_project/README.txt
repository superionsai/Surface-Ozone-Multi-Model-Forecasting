--> Run with xgb only model without tuning :-
Train shape: (1460, 14)
Test shape: (366, 14)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 5.8366731530069655
R2: 0.6436736918777134
PS C:\Users\saisi\OneDrive\Desktop\ASD390\ozone_ml_project> py main.py

Data ready for modeling.

Train shape: (1460, 14)
Test shape: (366, 14)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 5.8366731530069655
R2: 0.6436736918777134

Top Features:
 O3_lag1            0.551627
Leighton_ratio     0.112367
NO (µg/m³)         0.059372
Unnamed: 0         0.047043
NO2 (µg/m³)        0.030607
NH3 (µg/m³)        0.029923
VOC_NOx_ratio      0.028642
NOx (ppb)          0.026523
Benzene (µg/m³)    0.023935
PM2.5 (µg/m³)      0.022204
dtype: float32

Characteristics of current plots :-
from the folder ../ASD390/XGB/
-> Follows the overall trend well
-> Captures the medium variations
-> (0) Misses the sharp spikes (underestimating peaks)
XGBoost does not cover sudden pollution changes and regime shifts;
those which occur in seasonal events are also not cover and only average shifts are trained

Dominant features :- 
- O3_lag1 : 0.55
- Leighton_ratio : 0.11
- NO : 0.05
Leighton_ratio and NO are the features which we had already discovered from chemistry (as mentioned in report)
O3_lag1 is the measured feature itself from surroundings

Dropped column ["Unnamed: 0"], which contributed with 0.047 yet expected was 0 (time index trend)

_____

--> With CROSS VALIDATION with regular time series
- Decent improvement from 0.64 to 0.72 R2 score
Data ready for modeling.

Train shape: (1460, 15)
Test shape: (366, 15)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 5.024599398420157
R2: 0.7183654914034403

Running Time Series Cross Validation...

--- Fold 1 ---
MAE: 3.9264, R2: 0.6497

--- Fold 2 ---
MAE: 5.1565, R2: 0.5693

--- Fold 3 ---
MAE: 5.9805, R2: 0.7052

--- Fold 4 ---
MAE: 4.9043, R2: 0.7389

--- Fold 5 ---
MAE: 5.0617, R2: 0.6961

--- CROSS VALIDATION RESULTS ---
CV MAE: 5.0059
CV R2: 0.6719

Top Features:
 O3_lag1            0.630267
Leighton_ratio     0.094009
NO (µg/m³)         0.051733
NO2 (µg/m³)        0.038850
NH3 (µg/m³)        0.031977
Benzene (µg/m³)    0.023548
SO2 (µg/m³)        0.022526
NOx (ppb)          0.022232
PM2.5 (µg/m³)      0.020001
VOC_NOx_ratio      0.019609
dtype: float32

- Mulitple feature importance have been improved, considering this model to be a better forecast using cross feature validation
- Considers O3_lag1 to have strong temporal persistence

_____

--> Added new features and stabilized model according to chemistry and smoothing
- Heavy importance to a single new feature (O3_roll3)
This is a new mean feature of t-2, t-1, t and this has importance of 0.77 of 1
Target itself is included inside the feature, hence the values obtained as overfitting
Shifting this to t-3, t-2 and t-1

Train shape: (1459, 20)
Test shape: (365, 20)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 2.6082505766463604
R2: 0.9020983476225614

Running Time Series Cross Validation...

--- Fold 1 ---
MAE: 2.9105, R2: 0.8048

--- Fold 2 ---
MAE: 3.0681, R2: 0.8215

--- Fold 3 ---
MAE: 3.5069, R2: 0.8760

--- Fold 4 ---
MAE: 2.4419, R2: 0.9278

--- Fold 5 ---
MAE: 2.6184, R2: 0.8941

--- CROSS VALIDATION RESULTS ---
CV MAE: 2.9091
CV R2: 0.8648

Top Error Cases:
                               Actual  Predicted   Residual
1970-01-01 00:00:00.000001637   16.67  44.690025 -28.020025
1970-01-01 00:00:00.000001636   23.88  48.548042 -24.668042
1970-01-01 00:00:00.000001596   77.33  55.521404  21.808596
1970-01-01 00:00:00.000001631   84.62  63.815353  20.804647
1970-01-01 00:00:00.000001498   65.19  47.381405  17.808595
1970-01-01 00:00:00.000001663   43.80  29.117434  14.682566
1970-01-01 00:00:00.000001606   56.87  42.202068  14.667932
1970-01-01 00:00:00.000001595   69.17  55.718494  13.451506
1970-01-01 00:00:00.000001679   24.02  36.701233 -12.681233
1970-01-01 00:00:00.000001780   51.98  40.314739  11.665261

Top Features:
 O3_roll3              0.778861
Leighton_ratio        0.038993
O3_volatility         0.032469
O3_lag1               0.031794
NO2_O3_interaction    0.029437
NOx (ppb)             0.013880
NO2 (µg/m³)           0.013328
NO (µg/m³)            0.012498
NOx_lag1              0.008611
NH3 (µg/m³)           0.007887
dtype: float32

_____

Due to heavy dependence on O3_roll3,
Checking model without that column

Data ready for modeling.

Train shape: (1459, 19)
Test shape: (365, 19)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 2.9977456283046773
R2: 0.8903591759279824

Running Time Series Cross Validation...

--- Fold 1 ---
MAE: 3.0585, R2: 0.7916

--- Fold 2 ---
MAE: 2.9618, R2: 0.8273

--- Fold 3 ---
MAE: 5.1101, R2: 0.7770

--- Fold 4 ---
MAE: 2.4949, R2: 0.9252

--- Fold 5 ---
MAE: 3.0926, R2: 0.8787

--- CROSS VALIDATION RESULTS ---
CV MAE: 3.3436
CV R2: 0.8400

Top Error Cases:
                               Actual  Predicted   Residual
1970-01-01 00:00:00.000001595   69.17  45.022373  24.147627
1970-01-01 00:00:00.000001631   84.62  60.860634  23.759366
1970-01-01 00:00:00.000001636   23.88  46.582336 -22.702336
1970-01-01 00:00:00.000001596   77.33  55.289539  22.040461
1970-01-01 00:00:00.000001498   65.19  48.607079  16.582921
1970-01-01 00:00:00.000001663   43.80  30.441622  13.358378
1970-01-01 00:00:00.000001634   66.01  54.667759  11.342241
1970-01-01 00:00:00.000001560   44.02  32.870827  11.149173
1970-01-01 00:00:00.000001677   48.08  37.507385  10.572615
1970-01-01 00:00:00.000001809   20.91  31.407991 -10.497991

Top Features:
 O3_lag1               0.419516
high_NOx              0.184989
NO2_O3_interaction    0.101895
Leighton_ratio        0.070537
NO2 (µg/m³)           0.054768
O3_volatility         0.042502
NO (µg/m³)            0.041304
NOx (ppb)             0.019116
NH3 (µg/m³)           0.018941
VOC_NOx_ratio         0.011886
dtype: float32

Balanced model and works well with our chemistry expectations
--> CONCLUSION: Finalized base model with XGB
Heading to LSTM

_____

Data ready for modeling.

Train shape: (1459, 19)
Test shape: (365, 19)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 2.9977456283046773
R2: 0.8903591759279824

Running Time Series Cross Validation...

--- Fold 1 ---
MAE: 3.0585, R2: 0.7916

--- Fold 2 ---
MAE: 2.9618, R2: 0.8273

--- Fold 3 ---
MAE: 5.1101, R2: 0.7770

--- Fold 4 ---
MAE: 2.4949, R2: 0.9252

--- Fold 5 ---
MAE: 3.0926, R2: 0.8787

--- CROSS VALIDATION RESULTS ---
CV MAE: 3.3436
CV R2: 0.8400

Top Error Cases:
                               Actual  Predicted   Residual
1970-01-01 00:00:00.000001595   69.17  45.022373  24.147627
1970-01-01 00:00:00.000001631   84.62  60.860634  23.759366
1970-01-01 00:00:00.000001636   23.88  46.582336 -22.702336
1970-01-01 00:00:00.000001596   77.33  55.289539  22.040461
1970-01-01 00:00:00.000001498   65.19  48.607079  16.582921
1970-01-01 00:00:00.000001663   43.80  30.441622  13.358378
1970-01-01 00:00:00.000001634   66.01  54.667759  11.342241
1970-01-01 00:00:00.000001560   44.02  32.870827  11.149173
1970-01-01 00:00:00.000001677   48.08  37.507385  10.572615
1970-01-01 00:00:00.000001809   20.91  31.407991 -10.497991

Top Features:
 O3_lag1               0.419516
high_NOx              0.184989
NO2_O3_interaction    0.101895
Leighton_ratio        0.070537
NO2 (µg/m³)           0.054768
O3_volatility         0.042502
NO (µg/m³)            0.041304
NOx (ppb)             0.019116
NH3 (µg/m³)           0.018941
VOC_NOx_ratio         0.011886
dtype: float32

Running LSTM model...
Epoch 0, Loss: 1.0287
Epoch 5, Loss: 0.9512
Epoch 10, Loss: 0.8569
Epoch 15, Loss: 0.7338
Epoch 20, Loss: 0.6155
Epoch 25, Loss: 0.5704
Epoch 30, Loss: 0.5458
Epoch 35, Loss: 0.5199
Epoch 40, Loss: 0.5038
Epoch 45, Loss: 0.4813

--- LSTM PERFORMANCE ---
LSTM MAE: 8.2675
LSTM R2: 0.3352

--- HYBRID MODEL PERFORMANCE ---
Hybrid MAE: 3.7059
Hybrid R2: 0.8532


HYBRID MODEL WORKS HEAVILY ON XGB (0.8) and LSTM (0.2)

_____

Data ready for modeling.

Train shape: (1459, 19)
Test shape: (365, 19)

Model trained.

--- MODEL PERFORMANCE ---
MAE: 2.9977456283046773
R2: 0.8903591759279824

Running Time Series Cross Validation...

--- Fold 1 ---
MAE: 3.0585, R2: 0.7916

--- Fold 2 ---
MAE: 2.9618, R2: 0.8273

--- Fold 3 ---
MAE: 5.1101, R2: 0.7770

--- Fold 4 ---
MAE: 2.4949, R2: 0.9252

--- Fold 5 ---
MAE: 3.0926, R2: 0.8787

--- CROSS VALIDATION RESULTS ---
CV MAE: 3.3436
CV R2: 0.8400

Top Error Cases:
                               Actual  Predicted   Residual
1970-01-01 00:00:00.000001595   69.17  45.022373  24.147627
1970-01-01 00:00:00.000001631   84.62  60.860634  23.759366
1970-01-01 00:00:00.000001636   23.88  46.582336 -22.702336
1970-01-01 00:00:00.000001596   77.33  55.289539  22.040461
1970-01-01 00:00:00.000001498   65.19  48.607079  16.582921
1970-01-01 00:00:00.000001663   43.80  30.441622  13.358378
1970-01-01 00:00:00.000001634   66.01  54.667759  11.342241
1970-01-01 00:00:00.000001560   44.02  32.870827  11.149173
1970-01-01 00:00:00.000001677   48.08  37.507385  10.572615
1970-01-01 00:00:00.000001809   20.91  31.407991 -10.497991

Top Features:
 O3_lag1               0.419516
high_NOx              0.184989
NO2_O3_interaction    0.101895
Leighton_ratio        0.070537
NO2 (µg/m³)           0.054768
O3_volatility         0.042502
NO (µg/m³)            0.041304
NOx (ppb)             0.019116
NH3 (µg/m³)           0.018941
VOC_NOx_ratio         0.011886
dtype: float32

Running LSTM model...
Epoch 0, Loss: 1.0451
Epoch 5, Loss: 0.9649
Epoch 10, Loss: 0.8683
Epoch 15, Loss: 0.7499
Epoch 20, Loss: 0.6430
Epoch 25, Loss: 0.5795
Epoch 30, Loss: 0.5525
Epoch 35, Loss: 0.5240
Epoch 40, Loss: 0.5099
Epoch 45, Loss: 0.4913

--- LSTM PERFORMANCE ---
LSTM MAE: 8.1430
LSTM R2: 0.3449

--- STACKED MODEL PERFORMANCE ---
Hybrid MAE: 2.9922
Hybrid R2: 0.8988

Meta-model weights:
XGB weight: 1.1066
LSTM weight: -0.0089
Bias: -3.0720

--> LSTM weight ~ 0
LSTM learns no new signal apart from XGBoost i.e. the existing model is heavily feature-driven

FINAL CONCLUSION :- LSTM can be eliminated
Added SHAP analysis

______


From SHAP analysis plots :-
1. NO2_O3_interaction 🔥
2. Leighton_ratio 🔥
3. O3_lag1 🔥
4. NO, NO2, NOx
5. O3_volatility

Top 2 features are chemistry based meaning :
- the model works majorly on chemical equilibrium and reactions
- O3 ↔ NO2 equilibrium and N02_o3 interaction that covers the non linear term 
- O3_lag1 : implies temporal dependence but not dominant hence we can show no over-reliance to existing patterns



# 🌾 Maize Drought Detection — Project Map

**Read this file first.** It explains every folder and the full pipeline order.

---

## What this project does

Predicts maize drought risk (Healthy / Moderate / Severe) using 30 days of
weather data from 16 field sites across Anantapur, Kurnool, and Sri Sathya
Sai districts, Andhra Pradesh.

Best model: XGBoost — 82.3% accuracy, 0.743 Macro-F1, 0.931 ROC-AUC.

---

## Pipeline order (run once when setting up or adding new data)

```
STEP 1                          STEP 2                        STEP 3
step1_clean_raw_weather/   →   step2_generate_spei_labels/ →  src/features/
clean_weather.py                generate_spei_labels.py        sequence_builder.py
(raw CSVs → cleaned)            (cleaned → SPEI + labels)      (30-day windows)
                                                                      ↓
                                                              STEP 4: src/models/
                                                              train_xgboost.py etc.
                                                              (train 4 models)
                                                                      ↓
                                                              STEP 5: models/saved/
                                                              (saved model files)
                                                                      ↓
                                                              STEP 6: app/gradio_app.py
                                                              (farmer-facing prediction UI)
```

Steps 1–5 were already completed. You only re-run them if you add new field data.
Day-to-day: only Step 6 (the app) is used.

---

## Folder guide

```
maize_drought_detection\
│
├── app\
│   └── gradio_app.py           ⭐ THE APP  →  python app\gradio_app.py
│                                              open http://localhost:7860
│
├── step1_clean_raw_weather\
│   └── clean_weather.py        Run if adding new raw field data
│
├── step2_generate_spei_labels\
│   └── generate_spei_labels.py Run after step1 to compute SPEI + labels
│
├── data\
│   ├── raw\weather\            Original ERA5-Land downloads (never edit)
│   │   └── <site_name>\        One folder per field site (16 sites)
│   │       └── *_weather.csv
│   │
│   └── processed\weather\      Cleaned CSVs with SPEI column + labels
│       ├── *_with_spei.csv     One per field site — this is what the app uses
│       └── sequences\
│           └── pooled_sequences.npz   Combined training array (30,9) windows
│
├── src\                        All Python source code
│   ├── config.py               ⭐ Single source of truth — paths, features,
│   │                              thresholds, class labels. Everything reads from here.
│   ├── data_ingestion\
│   │   └── weather_loader.py   Reads and validates raw weather CSVs
│   ├── preprocessing\
│   │   ├── water_balance.py    Computes P - ET rolling water balance
│   │   ├── spei.py             Fits log-logistic SPEI per calendar month
│   │   ├── labels.py           Converts SPEI values to class labels
│   │   └── run_pipeline.py     Runs preprocessing end-to-end
│   ├── features\
│   │   └── sequence_builder.py Builds (n, 30, 9) rolling sequence arrays
│   ├── models\
│   │   ├── lstm_model.py       LSTM architecture (2-layer, 64 hidden)
│   │   ├── ml_models.py        XGBoost, Random Forest, Transformer architectures
│   │   ├── train_xgboost.py    Train XGBoost only
│   │   ├── train_random_forest.py
│   │   ├── train_transformer.py
│   │   ├── train_all_models.py Train all 4 at once
│   │   └── compare_all_models.py  Generates results/final_model_comparison.csv
│   ├── inference\
│   │   └── multi_predictor.py  Loads any model and predicts — used by the app
│   └── utils\
│       └── logger.py           Logging setup
│
├── models\saved\               Trained model files (do not edit)
│   ├── xgboost.joblib          ⭐ Best model — used in the app
│   ├── xgboost_metadata.json
│   ├── random_forest.joblib
│   ├── random_forest_metadata.json
│   ├── drought_lstm.pt
│   ├── drought_lstm_metadata.json
│   ├── transformer.pt
│   └── transformer_metadata.json
│
├── results\                    Evaluation outputs (for thesis/report)
│   ├── final_model_comparison.csv   All 4 models side by side
│   ├── best_model_summary.json
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_feature_importance.png
│   ├── xgboost_evaluation.csv
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_feature_importance.png
│   ├── transformer_confusion_matrix.png
│   └── transformer_training_curves.png
│
├── tests\                      Unit tests — run with: pytest tests\
├── logs\pipeline.log           Auto-generated run log
├── requirements.txt            Python packages needed
├── pyproject.toml              Package configuration
└── DEPLOY.md                   Deployment notes

```

---

## Model comparison (why XGBoost is in the app)

| Model       | Accuracy | Macro F1 | ROC-AUC | In app? |
|-------------|----------|----------|---------|---------|
| XGBoost     | 82.3%    | 0.743    | 0.931   | YES     |
| Random Forest | 81.5%  | 0.700    | 0.921   | No — kept for thesis |
| LSTM        | 79.1%    | 0.688    | —       | No — kept for thesis |
| Transformer | 78.7%    | 0.677    | 0.908   | No — kept for thesis |

All 4 models remain saved and loadable. Only XGBoost is shown in the app
to avoid confusing the end user.

---

## The 9 input features

| Feature | What it represents |
|---|---|
| Wind_Speed_10m_Mean_24h | Wind speed at 10m height |
| Temperature_Air_2m_Max_24h | Daily maximum air temperature (K) |
| Temperature_Air_2m_Mean_24h | Daily mean air temperature (K) |
| Temperature_Air_2m_Min_24h | Daily minimum air temperature (K) |
| Derived_Relative_Humidity_2m_Max_24h | Max relative humidity (%) |
| Derived_Relative_Humidity_2m_Min_24h | Min relative humidity (%) |
| Precipitation_Flux | Rainfall (mm/day) |
| ReferenceET_PenmanMonteith_FAO56 | Reference evapotranspiration (mm/day) |
| Solar_Radiation_Flux | Incoming solar radiation (W/m²) |

---

## Drought class labels (SPEI thresholds)

| Class    | SPEI range     |
|----------|---------------|
| Healthy  | SPEI ≥ 0      |
| Moderate | -1.5 ≤ SPEI < 0 |
| Severe   | SPEI < -1.5   |

---

## How to run the app

```cmd
cd D:\satellite_time_series\maize_drought_detection
python app\gradio_app.py
```

Open browser → http://localhost:7860
Upload any *_with_spei.csv from data\processed\weather\ to test.

---

## Safe to delete anytime

- Any `__pycache__\` folder
- Any `.ipynb_checkpoints\` folder
- `.pytest_cache\`
- `data\_DELETE_old_intermediate_csvs\` (already renamed for deletion)

## Never delete

`src\`  `app\`  `models\saved\`  `data\raw\`  `data\processed\`
`step1_clean_raw_weather\`  `step2_generate_spei_labels\`  `results\`  `tests\`

# 🌾 Agricultural Drought Detection
## Weather Time-Series Machine Learning Pipeline
### Rayalaseema, Andhra Pradesh, India | IIITDM Kurnool — APSCHE-Funded Project (DeepDroughtUAV)

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 👋 For Interns — Read This First

This project predicts whether a farm field is experiencing **drought** on any given day,
using only weather data. You do not need to visit the field — the system uses satellite
climate data downloaded from the internet to make predictions.

**What the system does, in simple terms:**

```
Last 30 days of weather data (temperature, rain, wind, humidity, etc.)
        ↓
Machine Learning Model (Transformer)
        ↓
"This field is experiencing SEVERE DROUGHT"
"Confidence: 87%"
```

**Who uses this:** Agricultural officers, farmers, and researchers in Rayalaseema, AP
who need early warning of drought conditions before crop damage occurs.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Region** | Anantapur, Kurnool, Sri Sathya Sai — Andhra Pradesh |
| **Field Sites** | 8 verified agricultural sites |
| **Data Period** | 30 Jan 1979 – 29 Jun 2026 (~47 years of daily weather) |
| **Weather Source** | ERA5-Land reanalysis (European Centre for Medium-Range Weather Forecasts) |
| **Label Method** | SPEI (Standardized Precipitation Evapotranspiration Index) |
| **Prediction** | Drought class: Healthy / Moderate / Severe |
| **Input Window** | Past 30 days of weather → predict current drought status |
| **Best Model** | Transformer (Macro F1: 0.6995, ROC-AUC: 0.9242) |
| **Related Module** | Module 2: UAV + Sentinel-2 image segmentation (separate repo, in progress) |

---

## 📍 Field Sites (8 verified sites)

| # | Village | District | ERA5-Land Grid Cell (lat, lon) |
|---|---------|----------|-------------------------------|
| 1 | Cholasamudram | Sri Sathya Sai | 13.8°N, 77.6°E |
| 2 | Kurnool – Balajivillas | Kurnool | 15.8°N, 78.1°E |
| 3 | Kurnool – Dinnedevarapadu | Kurnool | 15.8°N, 78.0°E |
| 4 | Kurnool – Pullaiah | Kurnool | 15.6°N, 78.1°E |
| 5 | Lepakshi | Anantapur | 13.8°N, 77.6°E |
| 6 | Peddakadubur | Sri Sathya Sai | 15.7°N, 77.3°E |
| 7 | Sira | Sri Sathya Sai | 13.8°N, 77.0°E |
| 8 | Somandepalle | Kurnool | 14.0°N, 77.6°E |

> **Coordinates note:** these are the ERA5-Land grid-cell centers actually used for each
> site (~9 km / 0.1° resolution) — the coordinate the weather data was pulled for, not a
> precise on-ground GPS pin.
>
> **8 unvisited sites were removed** from the original 16-site dataset (the `mudigubba*`
> and `talupala*` sites) to ensure only physically verified locations are used for training.

---

## 📡 Data Source — ERA5-Land

### What is ERA5-Land?

ERA5-Land is a **reanalysis dataset** produced by ECMWF (European Centre for
Medium-Range Weather Forecasts) through the **Copernicus Climate Data Store**.

**"Reanalysis" means:** scientists combined historical weather observations from
satellites, weather stations, radiosondes (weather balloons), ships, and aircraft
with a weather model to create a consistent, gap-free record of Earth's climate
going back to 1950.

**Why we use ERA5-Land (not local weather stations):**
- Free and publicly available
- Covers every location on Earth (no missing data)
- Consistent quality from 1950 to present, updated close to real-time
- ~9 km spatial resolution (fine enough for field-level analysis)
- Daily temporal resolution

### How to Download ERA5-Land Data

**Step 1:** Create a free account at the Copernicus Climate Data Store:
👉 https://cds.climate.copernicus.eu/user/register

**Step 2:** Install the CDS API client:
```bash
pip install cdsapi
```

**Step 3:** Create an API key file at `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY_HERE
```

**Direct dataset link (this is the exact dataset this project's weather data came from):**
👉 https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators-timeseries?tab=download

> This repo does not currently include a ready-to-run download script for new sites —
> the existing 8 sites' data is already included in `data/raw/weather/`. If you need to
> add a new site, download it manually from the CDS link above using the same variable
> set listed in the table below, matching the column names in an existing site's CSV.

---

## 📊 Input Features — Why 9 Parameters?

### What ERA5-Land actually gives us: 11 raw weather columns

Each site's raw CSV (`data/raw/weather/<site>/*.csv`) has **14 columns total**: a date
column, latitude, longitude, and these **11 weather variables**:

| # | Raw Column Name | What it measures | Unit |
|---|----------------|-----------------|------|
| 1 | Wind_Speed_10m_Mean_24h | Mean wind speed at 10m height | m/s |
| 2 | Temperature_Air_2m_Max_24h | Maximum daily temperature at 2m height | °C |
| 3 | Temperature_Air_2m_Mean_24h | Mean daily temperature | °C |
| 4 | Temperature_Air_2m_Min_24h | Minimum daily temperature | °C |
| 5 | Derived_Relative_Humidity_2m_Max_24h | Maximum relative humidity | % |
| 6 | Derived_Relative_Humidity_2m_Min_24h | Minimum relative humidity | % |
| 7 | Precipitation_Flux | Daily rainfall | mm |
| 8 | ReferenceET_PenmanMonteith_FAO56 | Reference evapotranspiration | mm |
| 9 | Solar_Radiation_Flux | Solar energy reaching the surface | J/m² |
| 10 | Vapour_Pressure_Mean_24h | Mean vapour pressure | hPa |
| 11 | Vapour_Pressure_Deficit_at_Maximum_Temperature | Vapour pressure deficit at daily max temp | hPa |

### Final 9 model input features

Features **1–9** above (Wind Speed through Solar Radiation) are used as model input.
Features **10–11** (the two vapour pressure columns) are present in the raw download
but are **not** used as model inputs in this pipeline — they were not selected as
predictors for the current models.

`src/config.py`'s `MODEL_INPUT_FEATURES` list is the single source of truth for exactly
which 9 columns the models are allowed to see:

```python
MODEL_INPUT_FEATURES = [
    "Wind_Speed_10m_Mean_24h",
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Min_24h",
    "Derived_Relative_Humidity_2m_Max_24h",
    "Derived_Relative_Humidity_2m_Min_24h",
    "Precipitation_Flux",
    "ReferenceET_PenmanMonteith_FAO56",
    "Solar_Radiation_Flux",
]
```

---

## ⚠️ Data Leakage — What It Is and Why It Matters Here

### What is data leakage?

Data leakage means the model accidentally **sees information it shouldn't have**
during training. It's like a student memorizing exam answers instead of learning
the subject — the student appears to perform well but fails on real tests.

### How it could happen in this specific project

Our drought labels (Healthy/Moderate/Severe) are **computed from** two of the raw
weather features — Precipitation and Reference ET:

```
Water_Balance (daily) = Precipitation − Reference ET
        ↓
WB30 = 30-day rolling sum of Water_Balance
        ↓
SPEI = WB30 standardized per calendar month
        ↓
Drought Label (Healthy / Moderate / Severe) = threshold on SPEI
```

`Water_Balance`, `WB30`, and `SPEI` are **intermediate columns** — they exist in the
processed data (`data/processed/weather/*_with_spei.csv`) purely so the labeling
process is auditable. If any of these three were fed into the model as input features:

```
Model sees SPEI directly
        ↓
SPEI already determines the label by a fixed formula
        ↓
Model just learns to read the label off SPEI, instead of learning the actual
weather → drought relationship
        ↓
Looks like ~100% accuracy in testing, but the model has learned nothing that
generalizes to a real forecasting scenario where you don't already have SPEI
```

### The fix — enforced in code, not just convention

`src/config.py` defines this as a structural guard, not a rule someone has to remember:

```python
LEAKAGE_COLUMNS = ["Water_Balance", "WB30", "SPEI"]

assert not set(MODEL_INPUT_FEATURES) & set(LEAKAGE_COLUMNS), (
    "MODEL_INPUT_FEATURES must never contain a leakage column."
)
```

`src/features/sequence_builder.py` (which builds the model's training windows) is the
**only** module allowed to read from `MODEL_INPUT_FEATURES`, and this assertion runs
every time the pipeline starts — so a future edit that accidentally adds `SPEI` back
into the feature list fails loudly instead of silently inflating the reported accuracy.

---

## 🏷️ Drought Labels — SPEI Method

### What is SPEI?

SPEI (Standardized Precipitation Evapotranspiration Index) is an internationally
accepted drought index developed by Vicente-Serrano et al. (2010).

**Simple explanation:**
```
Water Balance = Rainfall − Evapotranspiration

If Water Balance is very negative for many days in a row → Drought
If Water Balance is near zero or positive           → No Drought
```

**How SPEI is actually computed in this pipeline:**
```
Step 1 — Daily water balance
    WB = Precipitation (mm) − Reference ET (mm)

Step 2 — 30-day rolling sum
    WB30 = sum of the last 30 days of WB

Step 3 — Month-wise standardization (removes seasonal bias)
    For each calendar month separately (Jan, Feb, ... Dec), fit a
    log-logistic distribution to that month's WB30 values, then convert
    each value to a standard-normal score = SPEI.

    Why month-wise? November is naturally drier than July in Rayalaseema.
    Comparing a November reading to a July reading directly would be
    misleading. Instead, a November value is compared only against
    historical Novembers.

Step 4 — Classify (thresholds, inclusive on the lower bound)
    SPEI ≥  0.0            → Healthy   (0)
    -1.5 ≤ SPEI < 0.0       → Moderate  (1)
    SPEI < -1.5             → Severe    (2)
```

### Label validation

The per-month SPEI approach was cross-checked during development: two different
fields' SPEI series **independently** surfaced **2002** — a documented regional
drought year in Andhra Pradesh — as one of their top-5 driest years, without that
year being specifically engineered or targeted. That two unrelated fields agreed
on the same historical drought year was treated as a sanity check that the SPEI
computation is behaving correctly, rather than as a formal accuracy benchmark.

---

## 🤖 Models Compared

### What each model does

**XGBoost and Random Forest (tree-based models):**
```
30-day weather window
        ↓
Flatten: compute mean, std, min, max, trend for each of the 9 features
        ↓
45 statistical features
        ↓
Gradient Boosted Trees / Random Forest
        ↓
Drought class
```

**LSTM (Long Short-Term Memory):**
```
30-day weather window (sequence of 9 features per day)
        ↓
LSTM reads day-by-day, carries forward what it's learned matters
        ↓
Drought class
```

**Transformer (winner):**
```
30-day weather window (sequence of 9 features per day)
        ↓
Multi-head attention: looks at all 30 days at once, learns which
days are most important for the prediction
        ↓
Drought class
```

### Results (chronological test split)

All 4 models are trained on the **same chronological 70/15/15 train/val/test split**,
split per field then pooled — the test set is each field's most recent ~15% of days,
so the model is always evaluated on dates strictly after what it trained on.

| Model | Accuracy | Macro F1 | ROC-AUC | Severe F1 |
|-------|----------|----------|---------|-----------|
| **Transformer ⭐** | 81.1% | **0.6995** | 0.9242 | **0.5423** |
| XGBoost | 81.5% | 0.6950 | **0.9265** | 0.5062 |
| Random Forest | 80.9% | 0.6821 | 0.9141 | 0.4830 |
| LSTM | 79.5% | 0.6695 | 0.9084 | 0.4637 |

**LSTM was additionally evaluated with Leave-One-Field-Out (LOFO) cross-validation** —
train on 7 fields, test on the 1 held-out field, repeat for all 8 fields. This is a
stricter test of generalization to a genuinely new, unseen location rather than just
a future date at an already-known field. The LSTM's mean LOFO Macro F1 was **0.81**,
notably higher than its chronological-split score above. LOFO has not yet been run
for the other 3 models.

### Why Transformer wins (not XGBoost, despite higher accuracy)

```
XGBoost Accuracy    = 81.5%  >  Transformer Accuracy    = 81.1%
        BUT
XGBoost Severe F1    = 0.5062  <  Transformer Severe F1    = 0.5423
```

Severe drought is the **rarest class** (~8% of days). Accuracy is misleading for
rare classes: if 80% of days are Healthy, a model that always predicts "Healthy"
scores 80% accuracy while detecting zero droughts.

**Macro F1** gives equal weight to all 3 classes:
```
Macro F1 = (Healthy F1 + Moderate F1 + Severe F1) / 3
```

Transformer detects more severe droughts correctly. Missing a severe drought means
a farmer could lose an entire crop before anyone is warned — so Macro F1, and
specifically Severe-class F1, is the metric that matters most here, and Transformer
wins on both.

---

## 🗂️ Project Structure

```
satellite_time_series/                 ← Root folder
│
├── app/
│   └── gradio_app.py                  ← Web app (run this to use the system)
│
├── data/
│   ├── raw/weather/                   ← Original ERA5-Land CSVs
│   │   ├── cholasamudram/             ← One folder per field site
│   │   ├── kurnool_balaji_villas/
│   │   ├── kurnool_dinedevarapadu/
│   │   ├── kurnool_pulliah/
│   │   ├── lepakshi/
│   │   ├── Peddakadubur/
│   │   ├── sira/
│   │   └── Somandepalle/
│   │
│   └── processed/weather/             ← After adding SPEI drought labels
│       ├── cholasamudram_with_spei.csv
│       ├── kurnool_balaji_villas_with_spei.csv
│       └── ... (one per site + pooled_with_spei.csv)
│
├── models/saved/                      ← Trained ML models (ready to use)
│   ├── transformer.pt                 ← Best model ⭐
│   ├── transformer_metadata.json      ← Model config and results
│   ├── xgboost.joblib
│   ├── random_forest.joblib           ← tracked via Git LFS (192MB)
│   └── drought_lstm.pt
│
├── results/                           ← Evaluation outputs
│   ├── final_model_comparison.png     ← Bar chart comparing all 4 models
│   ├── confusion_matrix_all_models.png
│   └── (per-model evaluation CSVs and plots)
│
├── src/                                ← All Python source code
│   ├── config.py                       ← ⭐ Central config (paths, features, thresholds)
│   ├── preprocessing/                  ← Water balance, SPEI, label generation
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── train_random_forest.py
│   │   ├── train.py                    ← LSTM trainer
│   │   ├── train_transformer.py
│   │   └── compare_all_models.py       ← Generates the comparison table above
│   ├── features/
│   │   └── sequence_builder.py         ← Builds 30-day input windows
│   └── inference/
│       └── multi_predictor.py          ← Loads any model and predicts
│
├── tests/                              ← Unit tests
├── requirements.txt
└── pyproject.toml
```

---

## ⚙️ Setup for Interns

### Step 1 — Install Conda (if not installed)
Download Anaconda: 👉 https://www.anaconda.com/download

### Step 2 — Clone the repository
```bash
git clone https://github.com/hemanthkumar0406/satellite_time_series.git
cd satellite_time_series
```

### Step 3 — Create the environment
```bash
conda create -n drought python=3.10
conda activate drought
pip install -r requirements.txt
```

### Step 4 — Run the web app
The 8 sites' raw + processed weather data and all 4 trained models are already
included in this repo — you do **not** need to download or retrain anything to
use the app.
```bash
python app/gradio_app.py
# Open in browser: http://localhost:7860
```

### (Advanced) Reproducing the pipeline from scratch
If you want to regenerate SPEI labels or retrain models yourself:
```bash
python src/features/sequence_builder.py   # rebuild the 30-day training windows
python src/models/train_xgboost.py
python src/models/train_random_forest.py
python src/models/train.py                # LSTM
python src/models/train_transformer.py
python src/models/compare_all_models.py
```
> **Known issue:** the `step1_clean_raw_weather/` and `step2_generate_spei_labels/`
> folders are legacy scripts from an earlier version of this pipeline and are
> **not currently connected** to the data layout above (they read/write different
> paths). The actively-used label-generation pipeline is `src/preprocessing/`, but
> its `run_pipeline.py` entry point currently has an unresolved import bug. Until
> that's fixed, re-labeling from raw data requires manual intervention — ask before
> assuming either script "just works."

---

## 🌐 Live Demo

The web app accepts a 30-day weather CSV and returns:
- 🟢 **Healthy** — No drought stress
- 🟡 **Moderate Drought** — Consider irrigation
- 🔴 **Severe Drought** — Urgent action needed

---

## 🔬 Related Work

| Module | Description | Status |
|--------|-------------|--------|
| **Module 1 (this repo)** | Weather time-series drought prediction | ✅ Complete |
| Module 2 | UAV + Sentinel-2 SegFormer image segmentation | 🔧 In Progress |

---

## 📚 Key References

1. Vicente-Serrano et al. (2010) — SPEI drought index
2. Vaswani et al. (2017) — *Attention Is All You Need* (Transformer architecture)
3. Copernicus ERA5-Land: https://cds.climate.copernicus.eu

---

## 📖 Citation

```bibtex
@misc{drought_detection_2026,
  author      = {Hemanth Kumar},
  title       = {Agricultural Drought Detection — Module 1},
  year        = {2026},
  institution = {IIITDM Kurnool},
  note        = {DeepDroughtUAV, APSCHE-funded project — Rayalaseema, Andhra Pradesh}
}
```

---

## 📬 Contact

**Hemanth Kumar**
Project Assistant — IIITDM Kurnool
GitHub: https://github.com/hemanthkumar0406

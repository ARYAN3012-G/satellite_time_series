"""
Module 1 — config.py

Single source of truth for the whole pipeline. Every other module imports
its constants from here rather than redefining them locally. This is what
makes the leakage boundary (Section 3/9 of the v1 review) structural instead
of a convention someone has to remember: MODEL_INPUT_FEATURES is the only
list the sequence builder is allowed to read from, and it is defined once,
here, without the derived columns in it.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PROCESSED_WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
SEQUENCES_DIR = PROCESSED_WEATHER_DIR / "sequences"
MODELS_SAVED_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in (PROCESSED_WEATHER_DIR, SEQUENCES_DIR, MODELS_SAVED_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Fields expected under data/raw/weather/<field_name>/<field_name>_weather.csv
# Auto-discovered at ingestion time too (see Module 2), but listed here as
# the expected set for validation/logging purposes.
# Updated: removed unvisited villages (mudigubba*, talupala*);
# renamed pedhakadubur -> Peddakadubur, somandepalle -> Somandepalle
EXPECTED_FIELDS = [
    "cholasamudram",
    "kurnool_balaji_villas",
    "kurnool_dinedevarapadu",
    "kurnool_pulliah",
    "lepakshi",
    "Peddakadubur",
    "sira",
    "Somandepalle",
]

# ---------------------------------------------------------------------------
# Raw schema — required columns in every incoming weather CSV
# ---------------------------------------------------------------------------
DATE_COLUMN = "valid_time"  # confirmed from actual CDS export columns

# Real CDS exports have been observed using different names for the date
# column across export batches (e.g. "valid_time" instead of "Date"). The
# loader renames the first alias it finds to DATE_COLUMN before validation,
# so this list is the single place to add a new alias if another naming
# variant shows up in a future export.
DATE_COLUMN_ALIASES = ["Date", "valid_time", "date", "time"]

REQUIRED_RAW_COLUMNS = [
    DATE_COLUMN,
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

# ---------------------------------------------------------------------------
# Model input features — THE LEAKAGE BOUNDARY
# ---------------------------------------------------------------------------
# This is the ONLY feature list the sequence builder (Module 6) and the
# LSTM (Module 7) are allowed to read from. Water_Balance, WB30, and SPEI
# are intermediate/derived columns used purely to build the label and must
# never appear here — including them would let the model reverse-engineer
# the labeling rule instead of learning to forecast from raw weather.
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

# Columns that exist in processed data but are structurally forbidden as
# model input. Used by a guard/assertion in the sequence builder so this
# can never be silently violated by a future edit.
LEAKAGE_COLUMNS = ["Water_Balance", "WB30", "SPEI"]

assert not set(MODEL_INPUT_FEATURES) & set(LEAKAGE_COLUMNS), (
    "MODEL_INPUT_FEATURES must never contain a leakage column."
)

# ---------------------------------------------------------------------------
# Water balance / SPEI parameters
# ---------------------------------------------------------------------------
WB_ROLLING_WINDOW_DAYS = 30  # WB30: 30-day rolling accumulation of P - ET

# SPEI is standardized via a log-logistic distribution fit SEPARATELY for
# each calendar month (1-12), not one global fit and not a raw z-score.
# Water balance is seasonal and non-normal; per-month fitting is what makes
# a "moderately dry" reading comparable across a monsoon month and a dry
# month. See Section 4 of the v1 review for the rationale.
SPEI_FIT_PER_CALENDAR_MONTH = True
SPEI_MIN_YEARS_FOR_FIT = 10  # guard: warn if a field has too little history

# ---------------------------------------------------------------------------
# Drought classification — 3-class scheme, McKee et al. (1993) convention
# ---------------------------------------------------------------------------
# Designed directly this way (not derived by merging a 5-class scheme after
# the fact) because the 5-class Healthy/Mild boundary (0 / -0.1) was shown
# to be the smallest, noisiest gap in the scheme and the dominant source of
# error (Section 6/7 of the v1 review).
DROUGHT_CLASSES = ["Healthy", "Moderate", "Severe"]

LABEL_MAP = {cls: i for i, cls in enumerate(DROUGHT_CLASSES)}
INVERSE_LABEL_MAP = {i: cls for cls, i in LABEL_MAP.items()}

# Thresholds are inclusive on the lower bound of each band.
#   Healthy:  SPEI >= 0
#   Moderate: -1.5 <= SPEI < 0
#   Severe:   SPEI < -1.5
SPEI_THRESHOLDS = {
    "Healthy": 0.0,
    "Moderate": -1.5,
    # Severe = anything below the Moderate threshold
}


def classify_spei(spei_value: float) -> str:
    """Single source of truth for SPEI -> class label. Both the batch label
    generator (Module 5) and the live inference path (Module 9) call this
    so the rule can never drift between training and serving."""
    if spei_value >= SPEI_THRESHOLDS["Healthy"]:
        return "Healthy"
    elif spei_value >= SPEI_THRESHOLDS["Moderate"]:
        return "Moderate"
    else:
        return "Severe"


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH_DAYS = 60  # Updated: 60-day window captures monsoon cycles better
# Strict forecast framing: label for day t+1 uses ONLY weather from
# [t-59 ... t]. No same-day leakage of day t+1's weather into the window.

# Ablation study -- test all these sequence lengths on HPC:
SEQUENCE_LENGTH_ABLATION = [7, 14, 30, 60, 90]

# Lag label feature: the previous day's drought class (0/1/2) as an extra input.
# At prediction time (forecasting day t+1), day t's label is already KNOWN --
# it is NOT future information. This captures the 94.8% temporal autocorrelation
# that raw weather features alone cannot fully exploit.
USE_LAG_LABEL_FEATURE = True

# ---------------------------------------------------------------------------
# Train / validation / test split (chronological, primary evaluation)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# Leave-one-field-out (LOFO) -- secondary, stricter evaluation
# ---------------------------------------------------------------------------
RUN_LOFO_EVAL = True

# ---------------------------------------------------------------------------
# Model hyperparameters (v2 -- tuned to fix overfitting observed in v1)
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE = 64       # was 128; smaller = less overfitting
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.45         # was 0.3; higher = more regularization
LEARNING_RATE = 5e-4        # was 1e-3; slower convergence = better generalization
BATCH_SIZE = 128            # was 64; larger batch = smoother gradients
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Baseline (persistence)
# ---------------------------------------------------------------------------
RUN_PERSISTENCE_BASELINE = True

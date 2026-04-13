"""
PhysioNet 2012 - Preprocessing Pipeline
Transforms long-format physionet2012_timeseries.csv into:
  1. physionet2012_pivoted.csv  — wide format, one row per patient per hour (good for EDA)
  2. physionet2012_tensor.npz  — numpy arrays ready for GRU/LSTM (X, mask, y, record_ids)

Usage:
    python preprocess.py

Expects physionet2012_timeseries.csv in the same directory.
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "physionet2012_timeseries.csv"
PIVOTED_OUT = "physionet2012_pivoted.csv"
TENSOR_OUT  = "physionet2012_tensor.npz"

N_HOURS = 48   # ICU window
BIN_SIZE = 60  # minutes per bin

STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}

CLINICAL_VARS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "pH"
]  # 36 vars (dropping RecordID/Age/Gender/Height/ICUType/Weight from time-series)

# ── STEP 1: LOAD & CLEAN ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Raw shape: {df.shape}")

# Drop static descriptor rows (time == 0, these are demographics not time-series)
ts = df[~df["parameter"].isin(STATIC_PARAMS)].copy()
static = df[df["parameter"].isin(STATIC_PARAMS)].copy()

# Replace sentinel -1 values with NaN (dataset uses -1 for "not recorded")
ts["value"] = ts["value"].replace(-1.0, np.nan)

# Keep only the 36 known clinical variables (drops any noise/unknown params)
ts = ts[ts["parameter"].isin(CLINICAL_VARS)]

# Clip time to 0–2880 minutes (48 hours), drop anything outside window
ts = ts[(ts["time_minutes"] >= 0) & (ts["time_minutes"] <= 2880)]

# ── STEP 2: BIN TO HOURLY ─────────────────────────────────────────────────────
print("Binning to hourly intervals...")
ts["hour"] = (ts["time_minutes"] // BIN_SIZE).clip(0, N_HOURS - 1).astype(int)

# When multiple readings exist in the same hour, take the mean
ts_hourly = (
    ts.groupby(["RecordID", "hour", "parameter"])["value"]
    .mean()
    .reset_index()
)

# ── STEP 3: PIVOT ─────────────────────────────────────────────────────────────
print("Pivoting to wide format...")
pivoted = ts_hourly.pivot_table(
    index=["RecordID", "hour"],
    columns="parameter",
    values="value",
    aggfunc="mean"
).reset_index()
pivoted.columns.name = None

# Ensure all 36 clinical vars exist as columns (fill missing ones with NaN)
for var in CLINICAL_VARS:
    if var not in pivoted.columns:
        pivoted[var] = np.nan

# Reorder columns: RecordID, hour, then sorted clinical vars
pivoted = pivoted[["RecordID", "hour"] + sorted(CLINICAL_VARS)]

# ── STEP 4: ADD STATIC FEATURES & LABELS ──────────────────────────────────────
print("Adding demographics and labels...")

# Extract static features (one row per patient from the static rows)
static_wide = static.pivot_table(
    index="RecordID",
    columns="parameter",
    values="value",
    aggfunc="first"
)
static_wide.columns.name = None
# RecordID is already the index — reset it cleanly
if "RecordID" in static_wide.columns:
    static_wide = static_wide.drop(columns=["RecordID"])
static_wide = static_wide.reset_index()

# Replace -1 sentinel in static features too
for col in ["Age", "Gender", "Height", "ICUType", "Weight"]:
    if col in static_wide.columns:
        static_wide[col] = static_wide[col].replace(-1.0, np.nan)

# Extract labels and set info
labels = df.groupby("RecordID").agg(
    In_hospital_death=("In-hospital_death", "first"),
    set=("set", "first")
).reset_index()

pivoted = pivoted.merge(static_wide, on="RecordID", how="left")
pivoted = pivoted.merge(labels, on="RecordID", how="left")

# Save pivoted CSV for EDA
pivoted.to_csv(PIVOTED_OUT, index=False)
print(f"\n✅ Saved: {PIVOTED_OUT}")
print(f"   Shape: {pivoted.shape}  ({pivoted['RecordID'].nunique()} patients × up to 48 hours)")
print(f"   Columns: RecordID, hour, {len(CLINICAL_VARS)} clinical vars, 5 static features, label, set")

# ── STEP 5: BUILD MODEL-READY TENSOR ──────────────────────────────────────────
print("\nBuilding model-ready tensor...")

# Get all unique patients
all_patients = labels[labels["In_hospital_death"].notna()].copy()
record_ids = all_patients["RecordID"].values
y = all_patients["In_hospital_death"].values.astype(np.float32)

# Compute global mean/std for normalisation (from training set only — set A)
train_ids = all_patients[all_patients["set"] == "A"]["RecordID"].values
train_pivoted = pivoted[pivoted["RecordID"].isin(train_ids)]

means = {}
stds = {}
for var in CLINICAL_VARS:
    col_data = train_pivoted[var].dropna()
    means[var] = col_data.mean() if len(col_data) > 0 else 0.0
    stds[var]  = col_data.std()  if len(col_data) > 1 else 1.0
    if stds[var] == 0:
        stds[var] = 1.0

# Build tensors: X shape = (N_patients, 48, 36), mask shape = (N_patients, 48, 36)
N = len(record_ids)
n_vars = len(CLINICAL_VARS)
var_order = sorted(CLINICAL_VARS)

X    = np.zeros((N, N_HOURS, n_vars), dtype=np.float32)
mask = np.zeros((N, N_HOURS, n_vars), dtype=np.float32)

pivoted_indexed = pivoted.set_index(["RecordID", "hour"])

print(f"  Building tensors for {N} patients...")
for i, rid in enumerate(record_ids):
    if i % 1000 == 0:
        print(f"  {i}/{N}...")

    patient_rows = pivoted_indexed.xs(rid, level="RecordID") if rid in pivoted_indexed.index.get_level_values(0) else None

    for j, var in enumerate(var_order):
        # Get the 48-length series for this variable
        series = np.full(N_HOURS, np.nan)
        if patient_rows is not None and var in patient_rows.columns:
            for hour_idx, val in patient_rows[var].items():
                if 0 <= hour_idx < N_HOURS:
                    series[hour_idx] = val

        # Observation mask: 1 where actually observed
        obs_mask = (~np.isnan(series)).astype(np.float32)
        mask[i, :, j] = obs_mask

        # Forward-fill then backward-fill, then fill remaining with training mean
        filled = pd.Series(series).ffill().bfill().fillna(means[var]).values

        # Normalise
        X[i, :, j] = (filled - means[var]) / stds[var]

# Save as compressed numpy archive
np.savez_compressed(
    TENSOR_OUT,
    X=X,
    mask=mask,
    y=y,
    record_ids=record_ids,
    var_names=np.array(var_order),
    means=np.array([means[v] for v in var_order]),
    stds=np.array([stds[v] for v in var_order])
)

print(f"\n✅ Saved: {TENSOR_OUT}")
print(f"   X shape    : {X.shape}   (patients × hours × vars)")
print(f"   mask shape : {mask.shape}")
print(f"   y shape    : {y.shape}   | mortality rate: {y.mean():.1%}")
print(f"   var_names  : {var_order[:5]} ... (36 total)")

# ── STEP 6: EDA SUMMARY ───────────────────────────────────────────────────────
print("\n── EDA Summary ──────────────────────────────────────────────────────────")
print(f"Total patients : {N}")
print(f"Set breakdown  :")
print(labels["set"].value_counts().to_string())
print(f"\nMortality by set:")
print(labels.groupby("set")["In_hospital_death"].mean().apply(lambda x: f"{x:.1%}").to_string())
print(f"\nMissingness per variable (% of patient-hours with no observation):")
missingness = {v: float(mask[:, :, i].mean()) for i, v in enumerate(var_order)}
miss_df = pd.DataFrame.from_dict(missingness, orient="index", columns=["observed_rate"])
miss_df["missing_rate"] = 1 - miss_df["observed_rate"]
miss_df = miss_df.sort_values("missing_rate", ascending=False)
print(miss_df["missing_rate"].apply(lambda x: f"{x:.1%}").to_string())

print("\n── Done ─────────────────────────────────────────────────────────────────")
print(f"Load for modelling with:")
print(f"  data = np.load('{TENSOR_OUT}')")
print(f"  X, mask, y = data['X'], data['mask'], data['y']")
print(f"  # X and mask are both (N, 48, 36) — concatenate them: torch.cat([X, mask], dim=-1) → (N, 48, 72)")
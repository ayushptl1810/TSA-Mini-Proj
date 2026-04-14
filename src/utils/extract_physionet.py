"""
PhysioNet Computing in Cardiology Challenge 2012
ICU Mortality Prediction - Data Extraction Script

Run this from the root of the extracted dataset folder:
    python extract_physionet2012.py

Outputs:
    - physionet2012_flat.csv       : One row per patient, features as columns (wide format)
    - physionet2012_timeseries.csv : Raw time-series events (long format)
    - physionet2012_summary.xlsx   : Excel workbook with both sheets + outcome labels
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIG ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parents[2]
DATA_DIR    = ROOT / "data"
BASE_DATA   = ROOT / "predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0"

SETS = {
    "a": {"data_dir": BASE_DATA / "set-a", "outcomes_file": BASE_DATA / "Outcomes-a.txt"},
    "b": {"data_dir": BASE_DATA / "set-b", "outcomes_file": BASE_DATA / "Outcomes-b.txt"},
    "c": {"data_dir": BASE_DATA / "set-c", "outcomes_file": BASE_DATA / "Outcomes-c.txt"},
}

# The 37 variables in the dataset
DESCRIPTORS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]
TIME_SERIES_VARS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "pH"
]

AGGREGATIONS = ["mean", "min", "max", "std", "first", "last", "count"]

# ─── HELPERS ────────────────────────────────────────────────────────────────

def parse_patient_file(filepath):
    """Parse a single patient .txt file into a DataFrame."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Time"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                time_str, param, value = parts
                try:
                    value = float(value)
                except ValueError:
                    continue
                # Convert time HH:MM to minutes
                if ":" in time_str:
                    h, m = time_str.split(":")
                    time_minutes = int(h) * 60 + int(m)
                else:
                    time_minutes = -1  # static descriptor
                rows.append({"time_minutes": time_minutes, "parameter": param, "value": value})
    return pd.DataFrame(rows)


def extract_patient_features(filepath):
    """
    From one patient file, extract:
    - Static descriptors (age, gender, etc.)
    - Aggregated statistics for each time-series variable
    Returns a flat dict (one row for wide-format CSV).
    """
    record_id = int(Path(filepath).stem)
    df = parse_patient_file(filepath)

    if df.empty:
        return {"RecordID": record_id}

    features = {"RecordID": record_id}

    # Static descriptors (time == 0)
    static = df[df["time_minutes"] == 0]
    for _, row in static.iterrows():
        if row["parameter"] in DESCRIPTORS:
            features[row["parameter"]] = row["value"]

    # Time-series aggregations
    ts = df[df["time_minutes"] > 0]
    for var in TIME_SERIES_VARS:
        var_data = ts[ts["parameter"] == var]["value"]
        if var_data.empty:
            for agg in AGGREGATIONS:
                features[f"{var}_{agg}"] = np.nan
        else:
            features[f"{var}_mean"] = var_data.mean()
            features[f"{var}_min"] = var_data.min()
            features[f"{var}_max"] = var_data.max()
            features[f"{var}_std"] = var_data.std()
            features[f"{var}_first"] = var_data.iloc[0]
            features[f"{var}_last"] = var_data.iloc[-1]
            features[f"{var}_count"] = var_data.count()

    return features


def load_outcomes(outcomes_file):
    """Load the outcomes file into a DataFrame."""
    df = pd.read_csv(outcomes_file)
    df.columns = [c.strip() for c in df.columns]
    # Standard columns: RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death
    return df


def load_set(set_name, config):
    """Load all patients + outcomes for one set (a, b, or c)."""
    data_dir = config["data_dir"]
    outcomes_file = config["outcomes_file"]

    patient_files = sorted(glob.glob(os.path.join(str(data_dir), "*.txt")))
    print(f"\n[Set {set_name.upper()}] Found {len(patient_files)} patient files in '{data_dir}'")

    # --- Wide (flat) features
    flat_rows = []
    for i, fp in enumerate(patient_files):
        if i % 500 == 0:
            print(f"  Processing patient {i+1}/{len(patient_files)}...")
        flat_rows.append(extract_patient_features(fp))
    flat_df = pd.DataFrame(flat_rows)

    # --- Long (raw time-series)
    long_rows = []
    for fp in patient_files:
        record_id = int(Path(fp).stem)
        df = parse_patient_file(fp)
        df.insert(0, "RecordID", record_id)
        long_rows.append(df)
    long_df = pd.concat(long_rows, ignore_index=True)

    # --- Outcomes
    if os.path.exists(outcomes_file):
        outcomes_df = load_outcomes(outcomes_file)
        flat_df = flat_df.merge(outcomes_df, on="RecordID", how="left")
        long_df = long_df.merge(outcomes_df[["RecordID", "In-hospital_death"]], on="RecordID", how="left")
        print(f"  Outcomes merged. Mortality rate: {outcomes_df['In-hospital_death'].mean():.1%}")
    else:
        print(f"  WARNING: Outcomes file '{outcomes_file}' not found — skipping labels.")

    flat_df["set"] = set_name.upper()
    long_df["set"] = set_name.upper()

    return flat_df, long_df


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    all_flat = []
    all_long = []

    for set_name, config in SETS.items():
        if not os.path.exists(str(config["data_dir"])):
            print(f"[Set {set_name.upper()}] Directory '{config['data_dir']}' not found — skipping.")
            continue
        flat_df, long_df = load_set(set_name, config)
        all_flat.append(flat_df)
        all_long.append(long_df)

    if not all_flat:
        print("\nERROR: No data found. Make sure you run this script from the dataset root folder.")
        return

    combined_flat = pd.concat(all_flat, ignore_index=True)
    combined_long = pd.concat(all_long, ignore_index=True)

    # Reorder: put RecordID, static, set, outcome columns first
    priority_cols = ["RecordID", "set"] + DESCRIPTORS[1:]  # skip RecordID duplicate
    outcome_cols = [c for c in combined_flat.columns if c in
                    ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]]
    other_cols = [c for c in combined_flat.columns if c not in priority_cols + outcome_cols]
    combined_flat = combined_flat[priority_cols + outcome_cols + other_cols]

    # ─── Save CSV outputs
    combined_flat.to_csv(DATA_DIR / "physionet2012_flat.csv", index=False)
    print(f"\n✅ Saved flat CSV: {DATA_DIR / 'physionet2012_flat.csv'}  ({combined_flat.shape[0]} rows × {combined_flat.shape[1]} cols)")

    combined_long.to_csv(DATA_DIR / "physionet2012_timeseries.csv", index=False)
    print(f"✅ Saved long CSV: {DATA_DIR / 'physionet2012_timeseries.csv'}  ({combined_long.shape[0]} rows)")

    # ─── Save Excel workbook (summary sheet + both formats)
    print("\nBuilding Excel summary workbook...")
    with pd.ExcelWriter(DATA_DIR / "physionet2012_summary.xlsx", engine="openpyxl") as writer:

        # Sheet 1: Flat (wide) — full feature matrix
        combined_flat.to_excel(writer, sheet_name="Wide_Features", index=False)

        # Sheet 2: Variable summary statistics
        summary_rows = []
        for var in TIME_SERIES_VARS:
            col = f"{var}_mean"
            if col in combined_flat.columns:
                s = combined_flat[col].dropna()
                summary_rows.append({
                    "Variable": var,
                    "Description": _var_description(var),
                    "N_patients_with_data": int(combined_flat[f"{var}_count"].gt(0).sum()) if f"{var}_count" in combined_flat.columns else "",
                    "Mean": round(s.mean(), 3) if not s.empty else "",
                    "Std": round(s.std(), 3) if not s.empty else "",
                    "Min": round(s.min(), 3) if not s.empty else "",
                    "Max": round(s.max(), 3) if not s.empty else "",
                    "Missingness_%": round(combined_flat[col].isna().mean() * 100, 1),
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Variable_Summary", index=False)

        # Sheet 3: Outcome breakdown
        if "In-hospital_death" in combined_flat.columns:
            outcome_summary = combined_flat.groupby(["set", "In-hospital_death"]).size().reset_index(name="Count")
            outcome_summary["In-hospital_death"] = outcome_summary["In-hospital_death"].map({0: "Survived", 1: "Died"})
            outcome_summary.to_excel(writer, sheet_name="Outcome_Breakdown", index=False)

        # Sheet 4: Long time-series sample (first 5000 rows to keep Excel manageable)
        combined_long.head(5000).to_excel(writer, sheet_name="Timeseries_Sample_5k", index=False)

    print(f"✅ Saved Excel workbook: {DATA_DIR / 'physionet2012_summary.xlsx'}")
    print("\n── DONE ─────────────────────────────────────────────────────────")
    print(f"   Total patients : {combined_flat['RecordID'].nunique()}")
    if "In-hospital_death" in combined_flat.columns:
        print(f"   Overall mortality: {combined_flat['In-hospital_death'].mean():.1%}")
    print(f"   Feature columns: {len(other_cols)} time-series features + {len(DESCRIPTORS)-1} demographics")
    print(f"   Sets included  : {combined_flat['set'].unique().tolist()}")


def _var_description(var):
    desc = {
        "ALP": "Alkaline phosphatase (IU/L)",
        "ALT": "Alanine transaminase (IU/L)",
        "AST": "Aspartate transaminase (IU/L)",
        "Albumin": "Serum albumin (g/dL)",
        "BUN": "Blood urea nitrogen (mg/dL)",
        "Bilirubin": "Serum bilirubin (mg/dL)",
        "Cholesterol": "Serum cholesterol (mg/dL)",
        "Creatinine": "Serum creatinine (mg/dL)",
        "DiasABP": "Invasive diastolic arterial blood pressure (mmHg)",
        "FiO2": "Fractional inspired O2 (0-1)",
        "GCS": "Glasgow Coma Score (3-15)",
        "Glucose": "Serum glucose (mg/dL)",
        "HCO3": "Serum bicarbonate (mmol/L)",
        "HCT": "Haematocrit (%)",
        "HR": "Heart rate (bpm)",
        "K": "Serum potassium (mEq/L)",
        "Lactate": "Lactate (mmol/L)",
        "MAP": "Invasive mean arterial blood pressure (mmHg)",
        "MechVent": "Mechanical ventilation (0/1)",
        "Mg": "Serum magnesium (mmol/L)",
        "NIDiasABP": "Non-invasive diastolic ABP (mmHg)",
        "NIMAP": "Non-invasive mean ABP (mmHg)",
        "NISysABP": "Non-invasive systolic ABP (mmHg)",
        "Na": "Serum sodium (mEq/L)",
        "PaCO2": "Partial pressure of CO2 in arterial blood (mmHg)",
        "PaO2": "Partial pressure of O2 in arterial blood (mmHg)",
        "Platelets": "Platelet count (cells/nL)",
        "RespRate": "Respiration rate (bpm)",
        "SaO2": "Arterial oxygen saturation (%)",
        "SysABP": "Invasive systolic arterial blood pressure (mmHg)",
        "Temp": "Core body temperature (°C)",
        "TroponinI": "Cardiac troponin I (μg/L)",
        "TroponinT": "Cardiac troponin T (μg/L)",
        "Urine": "Urine output (mL)",
        "WBC": "White blood cell count (cells/nL)",
        "pH": "Arterial blood pH",
    }
    return desc.get(var, var)


if __name__ == "__main__":
    main()
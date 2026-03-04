import os
import re
import argparse
from pathlib import Path

import pandas as pd


def extract_demographics(mimic_path: Path, output_path: str):
    """
    Extract ICU stay metadata needed for linking tables.
    
    Only extracts columns actually used by OC-Distill:
    - ICUSTAY_ID, HADM_ID, SUBJECT_ID (for joining)
    - INTIME, OUTTIME (for filtering notes by time window)
    """
    print("Extracting demographics...")
    
    df_icustays = pd.read_csv(mimic_path / "ICUSTAYS.csv")
    df_icustays = df_icustays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]]
    
    print(f"  ICUSTAYS: {len(df_icustays)} rows")
    
    df_icustays.to_csv(output_path, index=False)
    print(f"  Saved {len(df_icustays)} ICU stays to {output_path}")


def extract_notes(mimic_path: Path, output_path: str):
    """
    Extract clinical notes (Nursing, Radiology categories).
    """
    print("Extracting clinical notes...")
    
    notes_df = pd.read_csv(mimic_path / "NOTEEVENTS.csv")
    print(f"  Total notes: {len(notes_df)}")
    
    # Filter to notes with CHARTTIME and HADM_ID
    notes_df = notes_df[notes_df['CHARTTIME'].notna()]
    notes_df = notes_df[notes_df['HADM_ID'].notna()]
    print(f"  After removing missing CHARTTIME/HADM_ID: {len(notes_df)}")
    
    # Filter to relevant categories
    categories = ["Nursing/other", "Nursing", "Radiology"]
    notes_df = notes_df[notes_df["CATEGORY"].isin(categories)]
    print(f"  After category filter ({categories}): {len(notes_df)}")
    
    # Select columns
    notes_df = notes_df[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"]].reset_index(drop=True)
    notes_df["HADM_ID"] = notes_df["HADM_ID"].astype('int')
    notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'])
    
    # Remove duplicates
    notes_df = notes_df.drop_duplicates(subset=['TEXT'], keep='first')
    notes_df = notes_df.drop_duplicates(subset=['CHARTTIME'], keep='last')
    print(f"  After removing duplicates: {len(notes_df)}")
    
    notes_df.to_csv(output_path, index=False)
    print(f"  Saved {len(notes_df)} notes to {output_path}")


def normalize_icd9(code_str):
    """Normalize ICD-9 code by adding decimal point."""
    code_str = str(code_str).strip()
    if '.' in code_str:
        return code_str
    if len(code_str) > 3:
        return code_str[:3] + '.' + code_str[3:]
    return code_str


# Pattern for main diagnosis codes (001-999)
MAIN_DX_RE = re.compile(r'^\d{3}(\.\d{1,2})?$')


def is_main_diagnosis(code):
    """Check if code is a main diagnosis (001-999 range)."""
    if not isinstance(code, str):
        return False
    if not MAIN_DX_RE.match(code):
        return False
    head = int(code.split('.')[0])
    return 1 <= head <= 999


def extract_diagnoses(mimic_path: Path, output_path: str):
    """
    Extract and normalize ICD-9 diagnosis codes.
    """
    print("Extracting diagnosis codes...")
    
    diagnosis_df = pd.read_csv(mimic_path / "DIAGNOSES_ICD.csv")
    print(f"  Total entries: {len(diagnosis_df)}")
    
    # Normalize codes
    diagnosis_df['ICD9_NORM'] = diagnosis_df['ICD9_CODE'].apply(normalize_icd9)
    
    # Filter to main diagnoses only
    diagnosis_df = diagnosis_df[diagnosis_df['ICD9_NORM'].apply(is_main_diagnosis)]
    print(f"  After filtering to main diagnoses: {len(diagnosis_df)}")
    
    # Select columns and remove duplicates
    diagnosis_df = diagnosis_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_NORM']]
    diagnosis_df = diagnosis_df.drop_duplicates()
    
    # Aggregate by admission
    diagnosis_df = diagnosis_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg(
        codes=('ICD9_NORM', lambda x: list(x.unique())),
        n_codes=('ICD9_NORM', 'nunique')
    ).reset_index()
    
    diagnosis_df.to_csv(output_path, index=False)
    print(f"  Saved {len(diagnosis_df)} admissions to {output_path}")
    print(f"  Average codes per admission: {diagnosis_df['n_codes'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract auxiliary tables from MIMIC-III"
    )
    parser.add_argument(
        "--mimic_path", type=str, required=True,
        help="Path to MIMIC-III CSV files directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data",
        help="Output directory (default: ../data)"
    )
    args = parser.parse_args()
    
    mimic_path = Path(args.mimic_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extract_demographics(mimic_path, str(output_dir / "demographic_table.csv"))
    extract_notes(mimic_path, str(output_dir / "notes_table.csv"))
    extract_diagnoses(mimic_path, str(output_dir / "diagnosis_table.csv"))
    
    print("\nDone! Created:")
    print(f"  {output_dir}/demographic_table.csv")
    print(f"  {output_dir}/notes_table.csv")
    print(f"  {output_dir}/diagnosis_table.csv")


if __name__ == "__main__":
    main()

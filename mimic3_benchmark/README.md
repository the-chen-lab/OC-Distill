# MIMIC-III Benchmark Data Preparation for OC-Distill

This folder contains a modified version of the [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) repository, adapted for the OC-Distill pipeline. We start from this benchmark and add additional processing including ICD ontology integration and auxiliary data such as clinical notes and diagnoses.

## Prerequisites

1. **MIMIC-III Access**: Credentialed access via [PhysioNet](https://physionet.org/content/mimiciii/)
2. **MIMIC-III CSV Files**: Download to a local directory
3. **Python Environment**: Set up the conda environment from the project root (see main README)

## Full Data Preparation Pipeline

### Step 1: Build the Base Benchmark

From within this `mimic3_benchmark` folder:

```bash
# Extract subjects from MIMIC-III CSVs
python -m mimic3benchmark.scripts.extract_subjects /path/to/mimic-iii/csvs ./data/root/

# Validate events
python -m mimic3benchmark.scripts.validate_events ./data/root/

# Extract episodes
python -m mimic3benchmark.scripts.extract_episodes_from_subjects ./data/root/

# Split train/test
python -m mimic3benchmark.scripts.split_train_and_test ./data/root/

# Create task-specific datasets
python -m mimic3benchmark.scripts.create_in_hospital_mortality ./data/root/ ./data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_length_of_stay ./data/root/ ./data/length-of-stay/
```

### Step 2: Create NPZ Data Files

```bash
# Create mortality data files
python mortality_benchmark_derivation.py \
    --data_dir ./data \
    --output_dir ../mimic3_benchmark_data \
    --hours 48h 72h 96h

# Create length-of-stay data files
python los_benchmark_derivation.py \
    --data_dir ./data \
    --output_dir ../mimic3_benchmark_data \
    --hours 48h 72h 96h
```
#### NPZ File Format

| Key | Type | Description |
|-----|------|-------------|
| `X` | `np.ndarray` | Time series array `[N, T, 24]` - 24 selected vital features |
| `y` | `np.ndarray` | Labels - binary for mortality, 0-9 for LOS |
| `stay_id` | `np.ndarray` | ICU stay IDs for linking with other tables |

### Step 3: Extract Auxiliary Tables

```bash
python extract_auxiliary_tables.py \
    --mimic_path /path/to/mimic-iii/csvs \
    --output_dir ../data
```

This creates:
- `../data/demographic_table.csv` - Patient demographics and ICU metadata
- `../data/notes_table.csv` - Clinical notes
- `../data/diagnosis_table.csv` - ICD-9 diagnosis codes

### Step 4: Download ICD-9 Ontology

Download manually from BioPortal:

1. Go to https://bioportal.bioontology.org/ontologies/ICD9CM
2. Click the "Downloads" tab
3. Download TTL format
4. Save as `ICD9CM.ttl` in the `oc-distill` root directory

### Step 5: BioClinicalBERT

The teacher model uses [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) for encoding clinical notes. The model and tokenizer are downloaded automatically from HuggingFace. If you need to pre-download the files, run the following command:
```bash
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"
```

## Output Structure

After completing all steps, the parent `oc-distill` directory should contain:

```
oc-distill/
├── data/
│   ├── demographic_table.csv
│   ├── notes_table.csv
│   └── diagnosis_table.csv
├── mimic3_benchmark_data/
│   ├── mortality_train_48h.npz
│   ├── mortality_test_48h.npz
│   ├── los_train_48h.npz
│   ├── los_test_48h.npz
│   └── ... (72h, 96h versions)
├── ICD9CM.ttl
└── ...
```
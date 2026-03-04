import numpy as np
import argparse
import os
import sys

sys.path.append(".")
sys.path.append("..")

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer


# Selected vital sign feature indices (24 features)
SELECTED_FEATURES = np.array([
    2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
    60, 61, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75
])


def select_features(X):
    """Select subset of vital sign features."""
    return X[:, :, SELECTED_FEATURES]


def main():
    parser = argparse.ArgumentParser(
        description="Create mortality prediction data files"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Path to benchmark data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../mimic3_benchmark_data",
        help="Output directory for NPZ files"
    )
    parser.add_argument(
        "--hours", type=str, nargs="+", default=["48h", "72h", "96h"],
        help="Time windows to process"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for hour in args.hours:
        print(f"\n=== Processing mortality {hour} ===")
        period = int(hour[:-1])
        
        train_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(args.data_dir, f"in-hospital-mortality/{hour}", "train"),
            listfile=os.path.join(args.data_dir, f"in-hospital-mortality/{hour}/train/listfile_{hour}.csv"),
            period_length=period
        )
        test_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(args.data_dir, f"in-hospital-mortality/{hour}", "test"),
            listfile=os.path.join(args.data_dir, f"in-hospital-mortality/{hour}/test/listfile_{hour}.csv"),
            period_length=period
        )
        
        discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero"
        )
        
        print("Loading train data...")
        train_raw = utils.load_data(
            train_reader, discretizer, normalizer=None, 
            small_part=False, return_names=False
        )
        print("Loading test data...")
        test_raw = utils.load_data(
            test_reader, discretizer, normalizer=None, 
            small_part=False, return_names=False
        )
        
        train_X = select_features(train_raw["data"][0])
        test_X = select_features(test_raw["data"][0])
        
        train_y = np.asarray(train_raw["data"][1]).astype(int)
        train_stay_id = np.asarray(train_raw["stay_ids"]).astype(int)
        
        test_y = np.asarray(test_raw["data"][1]).astype(int)
        test_stay_id = np.asarray(test_raw["stay_ids"]).astype(int)
        
        print(f"Train: {len(train_y)} samples, {train_y.sum()} positive")
        print(f"Test: {len(test_y)} samples, {test_y.sum()} positive")
        
        # Save as compressed NPZ files
        train_path = os.path.join(args.output_dir, f"mortality_train_{hour}.npz")
        test_path = os.path.join(args.output_dir, f"mortality_test_{hour}.npz")
        
        np.savez_compressed(train_path, X=train_X, y=train_y, stay_id=train_stay_id)
        np.savez_compressed(test_path, X=test_X, y=test_y, stay_id=test_stay_id)
        
        print(f"Saved to {args.output_dir}/mortality_{{train,test}}_{hour}.npz")


if __name__ == "__main__":
    main()

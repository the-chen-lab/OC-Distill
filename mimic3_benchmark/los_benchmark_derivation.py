import numpy as np
import argparse
import os
import random
import sys

sys.path.append(".")
sys.path.append("..")

from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import LengthOfStayReader
from mimic3models.preprocessing import Discretizer


# Selected vital sign feature indices (24 features)
SELECTED_FEATURES = np.array([
    2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
    60, 61, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75
])


def select_features(X):
    """Select subset of vital sign features."""
    return X[:, :, SELECTED_FEATURES]


def drain_batchgen(gen):
    """Extract all data from batch generator."""
    X_list, y_list, stayid_list = [], [], []
    for _ in range(gen.steps):
        out = gen.next(return_y_true=True)
        X, yb, yb_true = out["data"]
        sids = out["stay_id"]
        for i in range(X.shape[0]):
            X_list.append(X[i])
            y_list.append(yb[i])
            stayid_list.append(sids[i])
    
    return (
        np.array(X_list),
        np.array(y_list),
        np.array(stayid_list, dtype=np.int64)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create length-of-stay prediction data files"
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
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for hour in args.hours:
        print(f"\n=== Processing LOS {hour} ===")
        
        train_reader = LengthOfStayReader(
            dataset_dir=os.path.join(args.data_dir, "length-of-stay", "train"),
            listfile=os.path.join(args.data_dir, f"length-of-stay/train/listfile_{hour}.csv")
        )
        test_reader = LengthOfStayReader(
            dataset_dir=os.path.join(args.data_dir, "length-of-stay", "test"),
            listfile=os.path.join(args.data_dir, f"length-of-stay/test/listfile_{hour}.csv")
        )
        
        discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero"
        )
        
        print("Loading train data...")
        train_data_gen = utils.BatchGen(
            reader=train_reader,
            discretizer=discretizer,
            normalizer=None,
            partition="custom",
            batch_size=2000,
            steps=None,
            shuffle=False,
            return_names=True
        )
        
        print("Loading test data...")
        test_data_gen = utils.BatchGen(
            reader=test_reader,
            discretizer=discretizer,
            normalizer=None,
            partition="custom",
            batch_size=2000,
            steps=None,
            shuffle=False,
            return_names=True
        )
        
        train_X, train_y, train_stay_id = drain_batchgen(train_data_gen)
        test_X, test_y, test_stay_id = drain_batchgen(test_data_gen)
        
        train_X = select_features(train_X)
        test_X = select_features(test_X)
        
        print(f"Train: {len(train_y)} samples")
        print(f"Test: {len(test_y)} samples")
        
        # Save as compressed NPZ files
        train_path = os.path.join(args.output_dir, f"los_train_{hour}.npz")
        test_path = os.path.join(args.output_dir, f"los_test_{hour}.npz")
        
        np.savez_compressed(train_path, X=train_X, y=train_y, stay_id=train_stay_id)
        np.savez_compressed(test_path, X=test_X, y=test_y, stay_id=test_stay_id)
        
        print(f"Saved to {args.output_dir}/los_{{train,test}}_{hour}.npz")


if __name__ == "__main__":
    main()

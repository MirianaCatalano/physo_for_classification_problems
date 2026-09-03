import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold


# =========================
# PROJECT DIRECTORY
# =========================
import sys
# Project root directory
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
from utils.validate_input_utils import (
    validate_eq_to_test,
    validate_experiment_name,
    validate_seed,
    validate_n_folds,
    validate_splits_folder_name,
)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Configuration file
    config_path = (project_dir / "configs" / "03_split_dataset_config.json")


    # =========================
    # INPUT PARAMETERS
    # =========================
    with open(config_path, "r") as f:
        config = json.load(f)
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_name = validate_experiment_name(config["EXPERIMENT_NAME"])
    n_folds = validate_n_folds(config["N_FOLDS"])
    seed = validate_seed(config["SEED"])
    splits_folder_name = validate_splits_folder_name(config["SPLITS_FOLDER_NAME"])

    # numer of repeats to reach the desired number of folds (must be a multiple of 5)
    repeat = n_folds // 5


    # =========================
    # DIRECTORY
    # =========================
    base_dir = project_dir / "tests"


    # =========================
    # SPLIT DATASETS
    # =========================
    print()
    print("=" * 80)
    print(f"Processing {eq_title} (index {i_eq})")
    print("=" * 80)
    try:
        # Find experiments
        equation_dir = base_dir / eq_title.replace(".", "_")
        if not equation_dir.exists():
            raise ValueError(f"No generated datasets found for {eq_title}.")
        
        experiment_dir = equation_dir / experiment_name / "generated_dataset"
        if not experiment_dir.exists():
            raise ValueError(f"No experiment found: {experiment_dir}")

        # Load dataset
        dataset_path = experiment_dir / "dataset.csv"
        data = pd.read_csv(dataset_path)

        # Create split directory
        splits_dir = (equation_dir / experiment_name / splits_folder_name)
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Extract features and labels
        excluded_columns = ["y", "y_noisy"]
        excluded_columns += [
            column for column in data.columns if column.startswith("y_bin_")
        ]
        excluded_columns += [
            column for column in data.columns if column.startswith("y_noisy_bin_")
        ]
        feature_columns = [
            column for column in data.columns if column not in excluded_columns
        ]
        X = data[feature_columns]


        # =========================
        # REGRESSION SPLITS
        # =========================
        print("\n--- REGRESSION SPLITS ---")
        regression_dir = (splits_dir / "regression")
        if regression_dir.exists():
            print(f"WARNING: Regression split directory already exists. Files may be overwritten: {regression_dir}\n")
        clean_dir = (regression_dir / "clean")
        noisy_dir = (regression_dir / "noisy")
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        y_clean = data[["y"]]
        y_noisy = data[["y_noisy"]]

        # Fixing seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        split_id = 0
        for r in range(repeat):
            # Same K-fold for both clean and noisy labels because it does not depend on the labels
            kfold = KFold(n_splits=5, shuffle=True, random_state=seed + r)
            for train_idx, test_idx in kfold.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train_clean = y_clean.iloc[train_idx]
                y_test_clean = y_clean.iloc[test_idx]
                y_train_noisy = y_noisy.iloc[train_idx]
                y_test_noisy = y_noisy.iloc[test_idx]

                # Clean
                X_train.to_csv(clean_dir / f"X_train_fold{split_id}.csv", index=False)
                X_test.to_csv(clean_dir / f"X_test_fold{split_id}.csv", index=False)
                y_train_clean.to_csv(clean_dir / f"y_train_fold{split_id}.csv", index=False)
                y_test_clean.to_csv(clean_dir / f"y_test_fold{split_id}.csv", index=False)

                # Noisy
                X_train.to_csv(noisy_dir / f"X_train_fold{split_id}.csv", index=False)
                X_test.to_csv(noisy_dir / f"X_test_fold{split_id}.csv", index=False)
                y_train_noisy.to_csv(noisy_dir / f"y_train_fold{split_id}.csv", index=False)
                y_test_noisy.to_csv( noisy_dir / f"y_test_fold{split_id}.csv", index=False)

                split_id += 1
        print(f"  Created {split_id} regression splits.")


        # =========================
        # CLASSIFICATION SPLITS
        # =========================
        print("\n--- CLASSIFICATION SPLITS ---")
        classification_dir = (splits_dir / "classification")
        classification_dir.mkdir(parents=True, exist_ok=True)

        # Find binarization columns
        clean_columns = sorted([column for column in data.columns if column.startswith("y_bin_")])
        for clean_column in clean_columns:
            q_label = clean_column.replace("y_bin_", "")
            noisy_column = (f"y_noisy_bin_{q_label}")

            # Current Threshold
            split_name = f"thr_{q_label}"
            threshold_dir = (classification_dir / split_name)
            if threshold_dir.exists():
                print(f"WARNING: Classification split directory with threshold {q_label} already exists. Files may be overwritten: {threshold_dir}\n")
            print(f"  Creating classification splits for threshold {q_label}...")
            clean_dir = (threshold_dir / "clean")
            noisy_dir = (threshold_dir / "noisy")
            clean_dir.mkdir(parents=True, exist_ok=True)
            noisy_dir.mkdir(parents=True, exist_ok=True)

            y_clean = data[[clean_column]]
            y_noisy = data[[noisy_column]]

            # Fixing seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            
            # Stratified K-fold over clean labels (it depends on the labels)
            split_id = 0
            for r in range(repeat):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
                for train_idx, test_idx in skf.split(X, y_clean.iloc[:, 0]):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_train_clean = (y_clean.iloc[train_idx])
                    y_test_clean = (y_clean.iloc[test_idx])

                    # Clean
                    X_train.to_csv(clean_dir / f"X_train_fold{split_id}.csv", index=False)
                    X_test.to_csv(clean_dir / f"X_test_fold{split_id}.csv", index=False)
                    y_train_clean.to_csv(clean_dir / f"y_train_fold{split_id}.csv", index=False)
                    y_test_clean.to_csv(clean_dir / f"y_test_fold{split_id}.csv", index=False)

                    split_id += 1
                    
            # Fixing seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            
            # Stratified K-fold over noisy labels (it depends on the labels)
            split_id = 0
            for r in range(repeat):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
                for train_idx, test_idx in skf.split(X, y_noisy.iloc[:, 0]):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_train_noisy = (y_noisy.iloc[train_idx])
                    y_test_noisy = (y_noisy.iloc[test_idx])

                    # Noisy
                    X_train.to_csv(noisy_dir / f"X_train_fold{split_id}.csv", index=False)
                    X_test.to_csv(noisy_dir / f"X_test_fold{split_id}.csv", index=False)
                    y_train_noisy.to_csv(noisy_dir / f"y_train_fold{split_id}.csv", index=False)
                    y_test_noisy.to_csv(noisy_dir / f"y_test_fold{split_id}.csv", index=False)

                    split_id += 1
            print(f"  {split_name}: created {split_id} splits.\n")
            
        # Save split configuration
        split_config = {
            "equation_index": i_eq,
            "description": eq_title,
            "experiment_name": experiment_name,
            "splits_folder_name": splits_folder_name,
            "splits_dir": str(splits_dir),
            "n_folds": n_folds,
            "n_splits_per_repeat": 5,
            "n_repeats": repeat,
            "seed": seed,
            "regression": {
                "method": "KFold",
                "shuffle": True
            },
            "classification": {
                "method": "StratifiedKFold",
                "shuffle": True
            }
        }
        split_config_path = splits_dir / "split_config.json"
        with open(split_config_path, "w") as f:
            json.dump(split_config, f, indent=4)
        print(f"\nSplit configuration saved in: {split_config_path}")
        print("\nAll splits created.")
    except Exception as e:
        print(f"Error while processing {eq_title}: {e}")
        raise

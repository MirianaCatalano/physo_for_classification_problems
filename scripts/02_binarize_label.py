import json
from pathlib import Path
import pandas as pd


# =========================
# PROJECT DIRECTORY
# =========================
import sys
# Project root directory
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
from utils.validate_input_utils import (
    validate_eq_to_test,
    validate_quantile_threshold,
    validate_experiment_name
)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Configuration file
    config_path = project_dir / "configs" / "02_binarize_label_config.json"


    # =========================
    # INPUT PARAMETERS
    # =========================
    with open(config_path, "r") as f:
        config = json.load(f)
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_name = validate_experiment_name(config["EXPERIMENT_NAME"])
    quantile_threshold = [validate_quantile_threshold(q) for q in config["QUANTILE_THRESHOLD"]]


    # =========================
    # DIRECTORY
    # =========================
    base_dir = project_dir / "tests"


    # =========================
    # PROCESS THIS EXPERIMENT
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
        df = pd.read_csv(dataset_path)

        # Load dataset configuration
        dataset_config_path = experiment_dir / "dataset_config.json"
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)

        # Binarization
        binarization_dataset_config = dataset_config.get("binarization", {})
        for q in quantile_threshold:
            # Skip the binarization if the columns already exist
            q_label = int(q * 100)
            y_column = f"y_bin_{q_label}"
            noisy_y_column = f"y_noisy_bin_{q_label}"
            if y_column in df.columns and noisy_y_column in df.columns:
                print(f"  Quantile {q} already binarized, skipping.")
                continue

            # Compute thresholds
            threshold = df["y"].quantile(q)
            noisy_threshold = df["y_noisy"].quantile(q)

            # Create binary labels
            df[y_column] = (df["y"] >= threshold).astype(int)
            df[noisy_y_column] = (df["y_noisy"] >= noisy_threshold).astype(int)
            # Update dataset configuration for this quantile
            binarization_dataset_config[f"thr_{q_label}"] = {
                                "y_threshold": float(threshold),
                                "y_noisy_threshold": float(noisy_threshold),
                            }
            print(f"  Quantile {q}: y threshold = {threshold:.6g}, noisy y threshold = {noisy_threshold:.6g}")
            print(f"    y positive samples: {df[y_column].mean():.3f}")
            print(f"    noisy y positive samples: {df[noisy_y_column].mean():.3f}")
                

            # =========================
            # SAVE DATASET
            # =========================
            df.to_csv(
                dataset_path,
                index=False
            )
            print(f"  Dataset updated: {dataset_path}")
            
            # =====================================
            # UPDATE AND SAVE DATASET CONFIGURATION
            # =====================================
            dataset_config["binarization"] = binarization_dataset_config
            with open(dataset_config_path, "w") as f:
                json.dump(dataset_config, f, indent=4)
            print(f"  Dataset configuration updated: {dataset_config_path}")
            print("Dataset binarized.\n")

    except Exception as e:
        print(
            f"Error while processing {eq_title}: {e}"
        )
        raise

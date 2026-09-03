import random
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

# =========================
# PROJECT DIRECTORY
# =========================
import sys
# Project root directory
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from utils.validate_input_utils import (
    validate_n_samples,
    validate_noise_level,
    validate_seed,
    validate_eq_to_test,
    validate_experiment_name
)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Configuration file
    config_path = (project_dir / "configs" / "01_generate_dataset_config.json")


    # =========================
    # INPUT PARAMETERS
    # =========================
    with open(config_path, "r") as f:
        config = json.load(f)
    n_samples = validate_n_samples(config["N_SAMPLES"])
    noise_level = validate_noise_level(config["NOISE_LEVEL"])
    seed = validate_seed(config["SEED"])
    unit_names = config["UNIT_NAMES"]
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_name = validate_experiment_name(config["EXPERIMENT_NAME"])

    # =========================
    # FIXING SEED
    # =========================
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


    # =========================
    # DIRECTORIES
    # =========================
    base_dir = project_dir / "tests"
    base_dir.mkdir(parents=True, exist_ok=True)


    # =========================
    # GENERATION CONFIGURATION
    # =========================

    # Input parameters that determine the generated dataset
    generation_config = {
        "N_SAMPLES": n_samples,
        "NOISE_LEVEL": noise_level,
        "SEED": seed,
    }


    # =========================
    # CREATION OF DATASET
    # =========================
    print()
    print("=" * 80)
    print(f"Processing {eq_title} (index {i_eq})")
    print("=" * 80)
    try:
        # =========================
        # CREATE EXPERIMENT FOLDER
        # =========================
        output_dir = (base_dir / eq_title.replace(".", "_") / experiment_name / "generated_dataset")
        if output_dir.exists():
            print(f"WARNING: Experiment already exists. Files may be overwritten: {output_dir}\n")
        print(f"Creating new experiment: {experiment_name}")
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # =========================
        # GENERATE DATA
        # =========================
        problem = Feyn.FeynmanProblem(
            i_eq=i_eq,
            original_var_names=True
        )
        X, y = problem.generate_data_points(
            n_samples=n_samples
        )
        y = y.flatten()


        # =========================
        # ADD GAUSSIAN NOISE
        # =========================
        y_rms = np.sqrt((y ** 2).mean())
        epsilon = (noise_level * np.random.normal(0, y_rms, len(y)))
        y_noisy = np.abs(y + epsilon)


        # =========================
        # CREATE DATAFRAME
        # =========================
        df = pd.DataFrame(X.T, columns=problem.X_names)

        # Regression labels
        df["y"] = y
        df["y_noisy"] = y_noisy

        print("\n--- DATASET INFO ---")
        print(f"   Number of samples : {len(df)}")
        print(f"   Number of features: {len(problem.X_names)}")
        print(f"   Noise level       : {noise_level * 100}%")


        # =========================
        # SAVE DATASET
        # =========================
        dataset_path = output_dir / "dataset.csv"
        df.to_csv(dataset_path, index=False)
        print(f"\n   Dataset saved in: {dataset_path}")


        # =========================
        # SAVE UNIT TABLE
        # =========================
        X_units_dict = {
            var_name: {u: int(v) for u, v in zip(unit_names, var_unit)}
            for var_name, var_unit in zip(problem.X_names, problem.X_units)
        }
        y_units_dict = {
            u: int(v) for u, v in zip(unit_names, problem.y_units)
        }
        unit_table = pd.concat(
            [pd.DataFrame(X_units_dict), pd.DataFrame({"y": y_units_dict})],
            axis=1
        ).T
        unit_table.index.name = "Feature"
        unit_path = (output_dir/ "unit_table.csv")
        unit_table.to_csv(unit_path)
        print(f"   Unit table saved in: {unit_path}")


        # =========================
        # SAVE METADATA
        # =========================
        x_ranges = {}
        for name, low, high in zip(problem.X_names, problem.X_lows, problem.X_highs):
            x_ranges[name] = [float(low), float(high)]

        dataset_config = {
            "experiment_name": experiment_name,
            "equation_index": i_eq,
            "equation_name": problem.eq_name,
            "description": eq_title,
            "formula_sympy": str(problem.formula_sympy),
            "n_variables": len(problem.X_names),
            "n_samples": n_samples,
            "noise_level": noise_level,
            "seed": seed,
            "y_rms": float(y_rms),
            "x_ranges": x_ranges,
            "generation_config": generation_config,
            "output_dir": str(output_dir)
        }

        dataset_config_path = (output_dir / "dataset_config.json")
        with open(dataset_config_path, "w") as f:
            json.dump(dataset_config, f, indent=4)
        print(f"   Dataset config saved in: {dataset_config_path}")
        print("\nDataset generated.")

    except Exception as e:

        print(
            f"Error while processing "
            f"{eq_title}: {e}"
        )
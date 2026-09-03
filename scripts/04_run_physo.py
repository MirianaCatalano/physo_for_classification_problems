import copy
import json
import gc
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import physo
from physo.learn import monitoring
import logging
from joblib import Parallel, delayed

# =========================
# PROJECT DIRECTORY
# =========================
import sys
import importlib.util
# Project root directory
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
from utils.validate_input_utils import (
    validate_eq_to_test,
    validate_experiment_name,
    validate_seed,
    validate_splits_folder_name,
    validate_task
)

# =========================
# PHYSO PIPELINE
# =========================
def run_physo_on_fold(fold, seed, results_dir, data_dir, X_units, y_unit, 
             custom_run_config, fixed_consts, fixed_consts_units,
             free_consts_names, free_consts_units, op_names, epochs):
    logger.info(f"Running fold {fold}...\n\n")

    # =========================
    # REPRODUCIBILITY
    # =========================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # =========================
    # FOLD OUTPUT DIRECTORY
    # =========================
    fold_dir = results_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================
    # MONITORING
    # =========================
    save_path_training_curves = str(fold_dir / "training_curves.png")
    save_path_log = str(fold_dir / "training.log")
    run_logger     = lambda : monitoring.RunLogger(
                                    save_path = save_path_log,
                                    do_save = True
                                )
    run_visualiser = lambda : monitoring.RunVisualiser (
                                    epoch_refresh_rate = epochs - 1,
                                    save_path = save_path_training_curves,
                                    do_show   = False,
                                    do_prints = False,
                                    do_save   = True, 
                                    #draw_all_progs_fit = True
                                )
    # =========================
    # LOAD SPLIT
    # =========================
    X_train = pd.read_csv(data_dir / f"X_train_fold{fold}.csv")
    y_train = pd.read_csv(data_dir / f"y_train_fold{fold}.csv")
    X_names = X_train.columns.tolist()
    
    # =========================
    # PREPARE DATA FOR PHYSO
    # =========================
    X_train = X_train.to_numpy().T
    y_train = y_train.to_numpy().astype(np.float64).ravel()
    
    run_config = copy.deepcopy(custom_run_config)
    expr, _ = physo.SR(
                        X_train, 
                        y_train,
                        # Giving names of variables (for display purposes)
                        X_names = X_names,
                        # Associated physical units (ignore or pass zeroes if irrelevant)
                        X_units = X_units,
                        # Giving name of root variable (for display purposes)
                        y_name  = "y",
                        # Associated physical units (ignore or pass zeroes if irrelevant)
                        y_units = y_unit,
                        # Fixed constants
                        fixed_consts       =  fixed_consts,
                        fixed_consts_units =  fixed_consts_units,
                        # Free constants names (for display purposes)
                        free_consts_names = free_consts_names,
                        # Units of free constants
                        free_consts_units = free_consts_units,
                        # Symbolic operations that can be used to make f
                        op_names = op_names, 
                        get_run_logger     = run_logger,
                        get_run_visualiser = run_visualiser,
                        # Run config
                        run_config = run_config,
                        # Parallel mode (only available when running from python scripts, not notebooks)
                        parallel_mode = False,
                        # Number of iterations (epochs)
                        epochs = epochs,  
                    )
    # =========================
    # SAVE BEST EXPRESSION
    # =========================
    expr.save(str(fold_dir / "best_expression.pkl"))
    
    # =========================
    # CLEAN MEMORY
    # =========================
    plt.close('all')
    del X_train, y_train
    gc.collect()
    
    logger.info(f"\nFold {fold} completed.\n\n")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # =========================
    # INITIALIZE LOGGER
    # =========================
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)


    # =============================
    # IMPORT THE CONFIGURATION FILE
    # =============================
    config_path = project_dir / "configs" / "04_run_physo_config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Setting parameter
    n_jobs = config.N_JOBS
    seed = validate_seed(config.SEED)
    i_eq, eq_title = validate_eq_to_test(tuple(config.EQ_TO_TEST))
    experiment_name = validate_experiment_name(config.EXPERIMENT_NAME)
    splits_folder_name = validate_splits_folder_name(config.SPLITS_FOLDER_NAME)
    task, threshold = validate_task(config.TASK, config.THRESHOLD)
    data_type = config.DATA_TYPE
    
    # Run parameter
    op_names = config.OP_NAMES
    fixed_consts = config.FIXED_CONSTS
    fixed_consts_units = config.FIXED_CONSTS_UNITS
    free_consts_names = config.FREE_CONSTS_NAMES
    free_consts_units = config.FREE_CONSTS_UNITS
    epochs = config.EPOCHS

    # Learning parameter
    reward_configs = config.reward_config
    learning_config = copy.deepcopy(config.learning_config)
    free_const_opti_args = config.free_const_opti_args
    priors_config = config.priors_config
    cell_config = config.cell_config

    logger.info("Configuration parameters loaded successfully.")
    logger.info(f"Equation to test: {i_eq} - {eq_title}")
    logger.info(f"Experiments to process: {experiment_name}")
    logger.info(f"Splits folder names for each experiment: {splits_folder_name}")
    logger.info(f"Task: {task}")
    if threshold is not None:
        logger.info(f"Quantile threshold: {threshold}")
    logger.info(f"Operation used: \n{np.vstack(op_names)}")
    logger.info(f"Fixed constants: {fixed_consts}")
    logger.info(f"Fixed constants units: {fixed_consts_units}")
    logger.info(f"Free constants names: {free_consts_names}")
    logger.info(f"Free constants units: {free_consts_units}")
    logger.info(f"Epochs: {epochs}")


    # =========================
    # FOLDERS MANAGEMENT
    # =========================
    
    # Create the results folder if it doesn't exist and save the configuration parameters
    equation_dir = project_dir / "tests" / eq_title.replace(".", "_")
    splits_dir = (equation_dir / experiment_name / splits_folder_name)
    if not splits_dir.exists():
        raise ValueError(f"No splits found for {experiment_name}: {splits_dir}")

    # Check data directories for task
    task_dir = splits_dir / task
    if not task_dir.exists():
        raise ValueError(f"No splits found for task '{task}': {task_dir}")
    if task == "classification":
        task_dir = task_dir / threshold
        if not task_dir.exists():
            raise ValueError(f"No splits found for threshold '{threshold}' of task '{task}': {task_dir}")
    if data_type not in {"clean", "noisy"}:
        raise ValueError(f"Invalid DATA_TYPE '{data_type}'. Expected 'clean' or 'noisy'.")
    data_dir = task_dir / data_type
    if not data_dir.exists():
        raise ValueError(
            f"No '{data_type}' data found for task '{task}': {data_dir}"
        )

    logger.info(f"Creating results directory...")
    if task == "classification":
        results_prefix = f"results_{task}_{threshold}"
    else:
        results_prefix = f"results_{task}"
    results_dir = splits_dir / results_prefix
    if results_dir.exists():
        logger.warning(f"Results directory already exists and will be overwritten: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")


    # ==================================
    # DEFINITION OF REWARD CONFIGURATION
    # ==================================
    if task not in config.reward_config:
        raise ValueError(f"No reward configuration found for task '{task}'.")
    reward_config = reward_configs[task]
    learning_config['rewards_computer'] = physo.physym.reward.make_RewardsComputer(**reward_config)
    
    # ---------- FINAL CONFIG ----------
    custom_run_config = {
        "learning_config"      : learning_config,
        "reward_config"        : reward_config,
        "free_const_opti_args" : free_const_opti_args,
        "priors_config"        : priors_config,
        "cell_config"          : cell_config,
    }


    # =========================
    # SAVE PHYSO CONFIGURATION
    # =========================

    # Convert priors configuration into a JSON-serializable format
    priors_config_serializable = []
    for name, args in config.priors_config:
        if isinstance(args, dict):
            new_args = {
                key: (float(value) if isinstance(value, (np.float32, np.float64)) else value)
                for key, value in args.items()
            }
        else:
            new_args = args
        priors_config_serializable.append([name, new_args])

    experiment_config = {
        # Setting parameter
        "seed": seed,
        "experiment_name": experiment_name,
        "equation_index": i_eq,
        "description": eq_title,
        "task": task,
        "threshold": threshold,
        "data_type": data_type,
        "n_jobs": n_jobs,
        "splits_folder_name": splits_folder_name,
        "results_dir": str(results_dir),
        # PhySO configuration
        "MAX_LENGTH": config.MAX_LENGTH,
        "OP_NAMES": config.OP_NAMES,
        "FIXED_CONSTS": config.FIXED_CONSTS,
        "FIXED_CONSTS_UNITS": config.FIXED_CONSTS_UNITS,
        "FREE_CONSTS_NAMES": config.FREE_CONSTS_NAMES,
        "FREE_CONSTS_UNITS": config.FREE_CONSTS_UNITS,
        "EPOCHS": config.EPOCHS,
        # Reward configuration
        "reward_config": {
            "reward_function": reward_config["reward_function"].__name__,
            "zero_out_unphysical": reward_config["zero_out_unphysical"],
            "zero_out_duplicates": reward_config["zero_out_duplicates"],
            "keep_lowest_complexity_duplicate": reward_config["keep_lowest_complexity_duplicate"],
            "parallel_mode": reward_config["parallel_mode"],
            "n_cpus": reward_config["n_cpus"],
        },
        # Learning configuration
        "learning_config": {
            "batch_size": learning_config["batch_size"],
            "max_time_step": learning_config["max_time_step"],
            "n_epochs": learning_config["n_epochs"],
            "gamma_decay": learning_config["gamma_decay"],
            "entropy_weight": learning_config["entropy_weight"],
            "risk_factor": learning_config["risk_factor"],
            "observe_units": learning_config["observe_units"],
            "optimizer": {
            "name": "Adam",
            "lr": 0.0025,
            },
        },
        # Free constant optimization configuration
        "free_const_opti_args": free_const_opti_args,
        # Priors configuration
        "priors_config": priors_config_serializable,
        # RNN cell configuration
        "cell_config": cell_config,
    }

    # Save configuration as JSON
    config_output_path = results_dir / "physo_config.json"
    with open(config_output_path, "w") as f:
        json.dump(experiment_config, f, indent=4)
    logging.info(f"PhySO configuration saved in: {config_output_path}")


    # =========================
    # LOAD UNITS TABLE
    # =========================
    units_table_path = equation_dir / experiment_name / "generated_dataset" / "unit_table.csv"
    units_table = pd.read_csv(units_table_path)
    units_table = units_table.set_index("Feature")

    # Extract unit names
    units_names = units_table.columns.tolist()
    # Extract feature names: all rows except the target "y"
    feature_name_list = units_table.index[units_table.index != "y"].tolist()

    # Units of the input features
    X_units = (units_table.loc[feature_name_list, units_names].values.astype(np.float64))
    # Uunit of the target "y"
    y_unit = units_table.loc["y", units_names].values.astype(np.float64)

    logger.info(f"Units table loaded successfully. Shape: {X_units.shape}")
    logger.info(f"Units of the data: {units_names}")
    logger.info(f"Target unit vector: {y_unit}")

    # =========================
    # LOAD SPLIT CONFIGURATION
    # =========================
    split_config_path = splits_dir / "split_config.json"
    with open(split_config_path, "r") as f:
        split_config = json.load(f)
    n_folds = split_config["n_folds"]

    # =================================
    # RUN SYMBOLIC DISCOVERY OF FORMULA
    # =================================
    Parallel(n_jobs=n_jobs)(
        delayed(run_physo_on_fold)(fold, seed, results_dir, data_dir, X_units, y_unit, 
                                   custom_run_config, fixed_consts, fixed_consts_units,
                                   free_consts_names, free_consts_units, op_names, epochs) 
                            for fold in range(n_folds)
        )
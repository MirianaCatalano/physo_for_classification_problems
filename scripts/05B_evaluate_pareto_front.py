import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import physo
import random
import joblib
from sklearn.preprocessing import StandardScaler

# =========================
# PROJECT DIRECTORY
# =========================
import sys

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
from utils.validate_input_utils import (
    validate_eq_to_test,
    validate_experiment_name,
    validate_splits_folder_name,
    validate_task
)
from utils.analysis_utils import (
    optimal_threshold,
    ba,
    evaluate_formulas_on_dataset,
    get_safe_unique_expressions,
    initialize_model
)


# ==========================
# EVALUATION OF PARETO FRONT
# ==========================
def evaluate_pareto_front_on_fold(fold, data_dir, results_dir, task, model_list, seed):
    logger.info(f"Evaluating Pareto front on fold {fold}...")

    # =========================
    # FOLD DIRECTORY
    # =========================
    fold_dir = results_dir / f"fold_{fold}"
    
    # ===============================
    # CHECK EXISTING RESULTS
    # ===============================
    rewards_path = fold_dir / "rewards_pareto_front.csv"
    if rewards_path.exists():
        existing_results = pd.read_csv(rewards_path)
        existing_models = set(existing_results["model"])
        for mod_name in model_list:
            if mod_name in existing_models:
                print(f"WARNING: Results for model '{mod_name}' already exist in {rewards_path} and will be overwritten.")

    # =========================
    # LOAD SPLIT
    # =========================
    X_train = pd.read_csv(data_dir / f"X_train_fold{fold}.csv")
    X_test = pd.read_csv(data_dir / f"X_test_fold{fold}.csv")
    y_train = pd.read_csv(data_dir / f"y_train_fold{fold}.csv")
    y_test = pd.read_csv(data_dir / f"y_test_fold{fold}.csv")

    # =========================
    # PREPARE DATA FOR PHYSO
    # =========================
    X_train = X_train.to_numpy().T
    X_test = X_test.to_numpy().T
    y_train = y_train.to_numpy().astype(np.float64).ravel()
    y_test = y_test.to_numpy().astype(np.float64).ravel()
    
    # Convert to tensors for expression execution
    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)


    # =========================
    # LOAD PARETO EXPRESSIONS
    # =========================
    pareto_csv_path = fold_dir / "training_curves_pareto.csv"
    pareto_pkl_path = fold_dir / "training_curves_pareto.pkl"
    if not pareto_csv_path.exists():
        raise ValueError(f"Pareto front CSV not found for fold {fold}: {pareto_csv_path}")
    if not pareto_pkl_path.exists():
        raise ValueError(f"Pareto front PKL not found for fold {fold}: {pareto_pkl_path}")

    pareto_front_formulas = physo.read_pareto_csv(str(pareto_csv_path))
    pareto_front_formulas = [str(formula) for formula in pareto_front_formulas]
    pareto_front_programs = physo.read_pareto_pkl(str(pareto_pkl_path))
    logger.info(f"Fold {fold}: loaded {len(pareto_front_formulas)} Pareto front expressions.")
    
    # ===========================
    # EVALUATE PARETO EXPRESSIONS
    # ===========================
    X_train_pareto = evaluate_formulas_on_dataset(pareto_front_programs, X_train)
    X_test_pareto  = evaluate_formulas_on_dataset(pareto_front_programs, X_test)
    X_train_pareto = pd.DataFrame(X_train_pareto, columns=pareto_front_formulas)
    X_test_pareto = pd.DataFrame(X_test_pareto, columns=pareto_front_formulas)
    logger.debug(f"Symbolic features generated: train={X_train_pareto.shape}, test={X_test_pareto.shape}")
    

    # ====================================
    # REMOVE UNSAFE AND DUPLICATE FEATURES
    # ====================================
    safe_expressions = get_safe_unique_expressions(X_train_pareto.columns)
    X_train_pareto = X_train_pareto[safe_expressions]
    X_test_pareto = X_test_pareto[safe_expressions]
    logger.debug(f"Safe symbolic features: train={X_train_pareto.shape}, test={X_test_pareto.shape}")
    
    # ====================================
    # SCALING OF PARETO EXPRESSIONS
    # ====================================
    scaler = StandardScaler()
    X_train_pareto = pd.DataFrame(
        scaler.fit_transform(X_train_pareto),
        columns=X_train_pareto.columns,
        index=X_train_pareto.index
    )
    X_test_pareto = pd.DataFrame(
        scaler.transform(X_test_pareto),
        columns=X_test_pareto.columns,
        index=X_test_pareto.index
    )
    scaler_path = fold_dir / f"StandardScaler_on_pareto_front.pkl"
    joblib.dump(scaler, scaler_path)
    
    # ===============================
    # EVALUATE MODELS ON PARETO FRONT
    # ===============================
    rewards_pareto_front = []
    if task == "classification":
        confusion_matrix_pareto_front = []
    for mod_name in model_list:
        # =========================
        # REPRODUCIBILITY
        # =========================
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # =========================
        # FIT AND SAVE MODEL
        # =========================
        logger.info(f"Evaluating model {mod_name} on fold {fold}...")
        model = initialize_model(mod_name, seed, task)
        model.fit(X_train_pareto, y_train.numpy().ravel())
        model_path = fold_dir / f"{mod_name}_model_on_pareto_front.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved in: {model_path}")

        # =========================
        # COMPUTE PREDICTIONS
        # =========================
        y_pred_train = model.predict(X_train_pareto)
        y_pred_test = model.predict(X_test_pareto)
        if task == "regression":
            from physo.physym.reward import SquashedNRMSE
            # =========================
            # COMPUTE REWARD
            # =========================
            reward_train = SquashedNRMSE(y_train, y_pred_train).item()
            reward_test = SquashedNRMSE(y_test, y_pred_test).item()
            logger.info(f"Fold {fold} - {mod_name} - Train reward: {reward_train:.5f}")
            logger.info(f"Fold {fold} - {mod_name} - Test reward: {reward_test:.5f}")
            
            # =========================
            # STORE RESULTS
            # =========================
            rewards_pareto_front.append({
                "model": mod_name,
                "reward_train": reward_train,
                "reward_test": reward_test,
            })
        else:
            # =========================
            # BINARY PREDICTIONS
            # =========================
            y_pred_train = torch.tensor(y_pred_train, dtype=torch.float64)
            y_pred_test = torch.tensor(y_pred_test, dtype=torch.float64)
            tau_star, _ = optimal_threshold(y_train, y_pred_train)
            y_pred_train_binary = (y_pred_train >= tau_star).int()
            y_pred_test_binary = (y_pred_test >= tau_star).int()
            
            # =========================
            # CONFUSION MATRIX - TRAIN
            # =========================
            TP_train = int(((y_pred_train_binary == 1) & (y_train == 1)).sum())
            TN_train = int(((y_pred_train_binary == 0) & (y_train == 0)).sum())
            FP_train = int(((y_pred_train_binary == 1) & (y_train == 0)).sum())
            FN_train = int(((y_pred_train_binary == 0) & (y_train == 1)).sum())
    
            # =========================
            # CONFUSION MATRIX - TEST
            # =========================
            TP_test = int(((y_pred_test_binary == 1) & (y_test == 1)).sum())
            TN_test = int(((y_pred_test_binary == 0) & (y_test == 0)).sum())
            FP_test = int(((y_pred_test_binary == 1) & (y_test == 0)).sum())
            FN_test = int(((y_pred_test_binary == 0) & (y_test == 1)).sum())
            
            # =========================
            # COMPUTE BALANCED ACCURACY
            # =========================
            reward_train = ba(
                torch.tensor(TN_train, dtype=torch.float64),
                torch.tensor(FP_train, dtype=torch.float64),
                torch.tensor(FN_train, dtype=torch.float64),
                torch.tensor(TP_train, dtype=torch.float64)
            ).item()
            reward_test = ba(
                torch.tensor(TN_test, dtype=torch.float64),
                torch.tensor(FP_test, dtype=torch.float64),
                torch.tensor(FN_test, dtype=torch.float64),
                torch.tensor(TP_test, dtype=torch.float64)
            ).item()
            logger.info(f"Fold {fold} - {mod_name} - Train reward: {reward_train:.5f}")
            logger.info(f"Fold {fold} - {mod_name} - Test reward: {reward_test:.5f}")
            
            # =========================
            # STORE RESULTS
            # =========================
            rewards_pareto_front.append({
                # Model info
                "model": mod_name,
                "reward_train": reward_train,
                "reward_test": reward_test
            })
            
            confusion_matrix_pareto_front.append({
                # Model info
                "model": mod_name,
                # Training set
                "TP_train": TP_train,
                "TN_train": TN_train,
                "FP_train": FP_train,
                "FN_train": FN_train,
                # Test set
                "TP_test": TP_test,
                "TN_test": TN_test,
                "FP_test": FP_test,
                "FN_test": FN_test,
                # Decision threshold
                "tau_star": float(tau_star),
            })
            
    if task == "classification":
        confusion_matrix_pareto_front = pd.DataFrame(confusion_matrix_pareto_front)
        confusion_matrix_path = fold_dir / "confusion_matrix_pareto_front.csv"
        if confusion_matrix_path.exists():
            existing_confusion = pd.read_csv(confusion_matrix_path)
            existing_confusion = existing_confusion[~existing_confusion["model"].isin(model_list)]
            confusion_matrix_pareto_front = pd.concat([existing_confusion, confusion_matrix_pareto_front], ignore_index=True)

        confusion_matrix_pareto_front.to_csv(confusion_matrix_path, index=False)
        logger.info(f"Pareto front confusion matrix saved in: {confusion_matrix_path}")
    
    
    # ===============================
    # SAVE PARETO FRONT RESULTS
    # ===============================
    rewards_pareto_front = pd.DataFrame(rewards_pareto_front)
    rewards_path = fold_dir / "rewards_pareto_front.csv"
    if rewards_path.exists():
        existing_results = pd.read_csv(rewards_path)
        # Remove results for models that are being re-evaluated
        existing_results = existing_results[~existing_results["model"].isin(model_list)]
        rewards_pareto_front = pd.concat([existing_results, rewards_pareto_front], ignore_index=True)
    rewards_pareto_front.to_csv(rewards_path, index=False, float_format="%.4f")
    logger.info(f"Pareto front results saved in: {rewards_path}\n")
                
            
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
    
    
    # =========================
    # CONFIGURATION FILE
    # =========================
    config_path = project_dir / "configs" / "05_evaluate_expressions_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)


    # =========================
    # LOG CONFIGURATION
    # =========================
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_name = validate_experiment_name(config["EXPERIMENT_NAME"])
    splits_folder_name = validate_splits_folder_name(config["SPLITS_FOLDER_NAME"])
    task, threshold = validate_task(config["TASK"],config["THRESHOLD"])
    data_type = config["DATA_TYPE"]
    model_list = config["MODEL_LIST"]
    
    logging.info("Configuration parameters loaded successfully.")
    logging.info(f"Equation to test: {i_eq} - {eq_title}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Splits folder: {splits_folder_name}")
    logging.info(f"Task: {task}")
    if threshold is not None:
        logging.info(f"Quantile threshold: {threshold}")
    logging.info(f"Data type: {data_type}")
    logging.info(f"Model to use to predict from Pareto front expressions: {model_list}")


    # =========================
    # FOLDERS MANAGEMENT
    # =========================
    
    # Create the results folder if it doesn't exist and save the configuration parameters
    equation_dir = project_dir / "tests" / eq_title.replace(".", "_")
    experiment_dir = equation_dir / experiment_name
    splits_dir = experiment_dir / splits_folder_name
    # Check data directories for task
    task_dir = splits_dir / task
    if task == "classification":
        task_dir = task_dir / threshold
    if data_type not in {"clean", "noisy"}:
        raise ValueError(f"Invalid DATA_TYPE '{data_type}'. Expected 'clean' or 'noisy'.")
    data_dir = task_dir / data_type
    # Check results directory for task
    if task == "classification":
            results_prefix = f"results_{task}_{threshold}"
    else:
        results_prefix = f"results_{task}"
    results_dir = splits_dir / results_prefix
    if not results_dir.exists():
        raise ValueError(f"No results found: {results_dir}")

    logger.info(f"Equation directory: {equation_dir}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Splits directory: {splits_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Results directory: {results_dir}")
        
    
    # =========================
    # LOAD SPLIT CONFIGURATION
    # =========================
    split_config_path = splits_dir / "split_config.json"
    with open(split_config_path, "r") as f:
        split_config = json.load(f)
    n_folds = split_config["n_folds"]
    
    # =========================
    # LOAD PHYSO CONFIGURATION
    # =========================
    physo_config_path = results_dir / "physo_config.json"
    with open(physo_config_path, "r") as f:
        physo_config = json.load(f)
    seed = physo_config["seed"]
        
        
    # =========================
    # SAVE CONFIGURATION
    # =========================
    experiment_config = {
        # Experiment settings
        "seed": seed,
        "experiment_name": experiment_name,
        "equation_index": i_eq,
        "description": eq_title,
        "task": task,
        "threshold": threshold,
        "data_type": data_type,
        "splits_folder_name": splits_folder_name,
        "model_list": model_list,
        "results_dir": str(results_dir),
    }
    config_save_path = results_dir / "pareto_config.json"
    with open(config_save_path, "w") as f:
        json.dump(experiment_config, f, indent=4)
    logger.info(f"Configuration saved in: {config_save_path}")
        
        
    # =================================
    # EVALUATE PARETO EXPRESSIONS
    # =================================
    for fold in range(n_folds):
        evaluate_pareto_front_on_fold(fold, data_dir, results_dir, task, model_list, seed)
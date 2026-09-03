import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import physo

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
    validate_task,
    validate_metric
)
from utils.analysis_utils import (
    optimal_threshold,
    ba
)


# =========================
# EVALUATION OF EXPRESSIONS
# =========================
def evaluate_best_expression_on_fold(fold, data_dir, results_dir, task):
    logger.info(f"Evaluating best expression on fold {fold}...")

    # =========================
    # FOLD DIRECTORY
    # =========================
    fold_dir = results_dir / f"fold_{fold}"

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
    # LOAD BEST EXPRESSION
    # =========================
    expression_path = fold_dir / "best_expression.pkl"
    if not expression_path.exists():
        raise ValueError(f"Best expression not found for fold {fold}: {expression_path}")
    expr = physo.read_pareto_pkl(str(expression_path))

    # =========================
    # COMPUTE PREDICTIONS
    # =========================
    y_pred_train = expr.execute(X_train)
    y_pred_test = expr.execute(X_test)

    # =========================
    # EVALUATION
    # =========================
    if task == "regression":
        from physo.physym.reward import SquashedNRMSE
        # =========================
        # COMPUTE REWARD
        # =========================
        reward_train = SquashedNRMSE(y_train, y_pred_train).item()
        reward_test = SquashedNRMSE(y_test, y_pred_test).item()
        logger.info(f"Fold {fold} - Train reward: {reward_train:.5f}")
        logger.info(f"Fold {fold} - Test reward: {reward_test:.5f}")
    
    elif task == "classification":
        # =========================
        # COMPUTE OPTIMAL THRESHOLD
        # =========================
        tau_star, _ = optimal_threshold(y_train, y_pred_train)
        logger.debug(f"Fold {fold} - Optimal threshold: {tau_star:.5f}")

        # =========================
        # BINARY PREDICTIONS
        # =========================
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
        logger.info(f"Fold {fold} - Train reward: {reward_train:.5f}")
        logger.info(f"Fold {fold} - Test reward: {reward_test:.5f}")
        
        confusion_matrix = pd.DataFrame({
                # Training set
                "TP_train": [TP_train],
                "TN_train": [TN_train],
                "FP_train": [FP_train],
                "FN_train": [FN_train],
                # Test set
                "TP_test": [TP_test],
                "TN_test": [TN_test],
                "FP_test": [FP_test],
                "FN_test": [FN_test],
                # Decision threshold
                "tau_star": [float(tau_star)],
            })
        confusion_matrix_path = fold_dir / "confusion_matrix_best_expr.csv"
        confusion_matrix.to_csv(confusion_matrix_path, index=False)
        logger.info(f"Confusion matrix saved in: {confusion_matrix_path}")
        
    # =========================
    # SAVE RESULTS
    # =========================
    rewards = pd.DataFrame({
        "reward_train": [reward_train],
        "reward_test": [reward_test],
    })
    rewards_path = fold_dir / "rewards_best_expr.csv"
    rewards.to_csv(rewards_path, index=False, float_format="%.4f")
    logger.info(f"Rewards saved in: {rewards_path}")
    logger.info(f"Fold {fold} completed.\n")


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
    
    logging.info("Configuration parameters loaded successfully.")
    logging.info(f"Equation to test: {i_eq} - {eq_title}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Splits folder: {splits_folder_name}")
    logging.info(f"Task: {task}")
    if threshold is not None:
        logging.info(f"Quantile threshold: {threshold}")
    logging.info(f"Data type: {data_type}")


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
        
        
    # =================================
    # EVALUATE BEST EXPRESSIONS
    # =================================
    for fold in range(n_folds):
        evaluate_best_expression_on_fold(fold, data_dir, results_dir, task)
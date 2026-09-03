import pandas as pd
from pathlib import Path
import logging
import json

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
    validate_splits_folder_name
)
from utils.analysis_utils import (
    find_available_results,
    load_results,
    compute_metrics,
    create_results_summary,
    create_statistics_summary
)


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
    config_path = project_dir / "configs" / "06_results_summary_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    
    # =========================
    # LOG CONFIGURATION
    # =========================
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_name = validate_experiment_name(config["EXPERIMENT_NAME"])
    splits_folder_name = validate_splits_folder_name(config["SPLITS_FOLDER_NAME"])
    data_type = config["DATA_TYPE"]
    if data_type not in {"clean", "noisy"}:
        raise ValueError(f"Invalid DATA_TYPE '{data_type}'. Expected 'clean' or 'noisy'.")
    logger.info("Configuration parameters loaded successfully.")
    logger.info(f"Equation to test: {i_eq} - {eq_title}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Splits folder: {splits_folder_name}")
    logger.info(f"Data type: {data_type}")


    # =========================
    # FOLDERS MANAGEMENT
    # =========================
    
    # Create the results folder if it doesn't exist and save the configuration parameters
    equation_dir = project_dir / "tests" / eq_title.replace(".", "_")
    experiment_dir = equation_dir / experiment_name
    splits_dir = experiment_dir / splits_folder_name
    logger.info(f"Equation directory: {equation_dir}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Splits directory: {splits_dir}")
        
    
    # =========================
    # LOAD SPLIT CONFIGURATION
    # =========================
    split_config_path = splits_dir / "split_config.json"
    if not splits_dir.exists():
        raise ValueError(f"Splits directory not found: {splits_dir}")
    with open(split_config_path, "r") as f:
        split_config = json.load(f)
    n_folds = split_config["n_folds"]
        
        
    # ============================
    # LOAD RESULTS
    # ============================
    available_results = find_available_results(experiment_dir, splits_dir, data_type)
    print("\n========== AVAILABLE RESULTS ==========")
    print(available_results)
    
    rewards_best_expr_list = []
    confusion_matrices_best_expr_list = []
    rewards_pareto_list = []
    confusion_matrices_pareto_list = []
    for _, result_config in available_results.iterrows():
        results_dir = result_config["results_dir"]
        
        # =====================================
        # LOAD EXPRESSION
        # =====================================
        rewards_best_expr = load_results(results_dir, n_folds, filename="rewards_best_expr.csv")
        
        # Add configuration information
        rewards_best_expr["noise"] = result_config["noise"]
        rewards_best_expr["task"] = result_config["task"]
        rewards_best_expr["imbalance"] = result_config["imbalance"]
        rewards_best_expr["threshold"] = result_config["threshold"]
        rewards_best_expr["data_type"] = result_config["data_type"]
        rewards_best_expr_list.append(rewards_best_expr)
        
        # Confusion matrices only for classification
        if result_config["task"] == "classification":
            confusion_matrices_best_expr = load_results(results_dir, n_folds, filename="confusion_matrix_best_expr.csv")
            confusion_matrices_best_expr["noise"] = result_config["noise"]
            confusion_matrices_best_expr["task"] = result_config["task"]
            confusion_matrices_best_expr["imbalance"] = result_config["imbalance"]
            confusion_matrices_best_expr["threshold"] = result_config["threshold"]
            confusion_matrices_best_expr["data_type"] = result_config["data_type"]
            confusion_matrices_best_expr_list.append(confusion_matrices_best_expr)

        
        # =====================================
        # LOAD PARETO FRONT
        # =====================================
        rewards_pareto = load_results(results_dir, n_folds, filename="rewards_pareto_front.csv")
        
        # Add configuration information
        rewards_pareto["noise"] = result_config["noise"]
        rewards_pareto["task"] = result_config["task"]
        rewards_pareto["imbalance"] = result_config["imbalance"]
        rewards_pareto["threshold"] = result_config["threshold"]
        rewards_pareto["data_type"] = result_config["data_type"]
        rewards_pareto_list.append(rewards_pareto)
        
        # Confusion matrices only for classification
        if result_config["task"] == "classification":
            confusion_matrices_pareto = load_results(results_dir, n_folds, filename="confusion_matrix_pareto_front.csv")
            confusion_matrices_pareto["noise"] = result_config["noise"]
            confusion_matrices_pareto["task"] = result_config["task"]
            confusion_matrices_pareto["imbalance"] = result_config["imbalance"]
            confusion_matrices_pareto["threshold"] = result_config["threshold"]
            confusion_matrices_pareto["data_type"] = result_config["data_type"]
            confusion_matrices_pareto_list.append(confusion_matrices_pareto)
        
    rewards_best_expr = pd.concat(rewards_best_expr_list, ignore_index=True)
    confusion_matrices_best_expr = pd.concat(confusion_matrices_best_expr_list, ignore_index=True)
    rewards_pareto = pd.concat(rewards_pareto_list, ignore_index=True)
    confusion_matrices_pareto = pd.concat(confusion_matrices_pareto_list, ignore_index=True)
    metrics_best_expr = compute_metrics(confusion_matrices_best_expr)
    metrics_pareto = compute_metrics(confusion_matrices_pareto) 
    
    results_summary = create_results_summary(rewards_best_expr, metrics_best_expr, rewards_pareto, metrics_pareto)
    logger.info("\n========== RESULTS SUMMARY ==========")
    logger.info(results_summary)
    summary_path = splits_dir / "results_summary.csv"
    results_summary.to_csv(summary_path, index=False, float_format="%.4f")
    logger.info(f"\nResults summary saved to: {summary_path}")
    
    statistics_summary = create_statistics_summary(results_summary)
    statistics_path = splits_dir / "statistics_summary.csv"
    statistics_summary.to_csv(statistics_path, index=False, float_format="%.4f")
    logger.info(f"\nStatistics summary saved to: {statistics_path}")

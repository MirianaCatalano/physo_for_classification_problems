from pathlib import Path
import logging
import json
import physo
from collections import Counter

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
    plot_expression_frequency,
    simplify_expr_str,
    get_safe_unique_expressions
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
    config_path = project_dir / "configs" / "07A_plot_histograms_config.json"
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
    n_digits = config["N_DIGITS"]
    logging.info("Configuration parameters loaded successfully.")
    logging.info(f"Equation to test: {i_eq} - {eq_title}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Splits folder: {splits_folder_name}")
    logging.info(f"Number of digits for expression simplification: {n_digits}")
    logging.info(f"Data type: {data_type}")


    # =========================
    # FOLDERS MANAGEMENT
    # =========================
    
    # Create the results folder if it doesn't exist and save the configuration parameters
    equation_dir = project_dir / "tests" / eq_title.replace(".", "_")
    experiment_dir = equation_dir / experiment_name
    splits_dir = experiment_dir / splits_folder_name
    plots_dir = splits_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Equation directory: {equation_dir}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Splits directory: {splits_dir}")
    logger.info(f"Plots directory: {plots_dir}")
        
    
    # =========================
    # LOAD SPLIT CONFIGURATION
    # =========================
    if not splits_dir.exists():
        raise ValueError(f"Splits directory not found: {splits_dir}")
    split_config_path = splits_dir / "split_config.json"
    with open(split_config_path, "r") as f:
        split_config = json.load(f)
    n_folds = split_config["n_folds"]
    
    
    # ============================
    # LOAD RESULTS
    # ============================
    available_results = find_available_results(experiment_dir, splits_dir, data_type)
    logger.info("\n========== AVAILABLE RESULTS ==========")
    logger.info(available_results)
    
    
    # =========================
    # PLOT EXPRESSION FREQUENCIES
    # =========================
    for _, result in available_results.iterrows():
        results_dir = Path(result["results_dir"])
        noise = result["noise"]
        task = result["task"]
        imbalance = result["imbalance"]
        threshold = result["threshold"]
        logger.info(f"Plotting results from: {results_dir}")       
        if task == "classification":
            plot_prefix = f"classification_thr_{int(threshold)}"
        else:
            plot_prefix = "regression"

        # =========================
        # BEST EXPRESSIONS
        # =========================
        best_expr_counter = Counter()
        for fold in range(n_folds):
            fold_dir = results_dir / f"fold_{fold}"
            expression_path = fold_dir / "best_expression.pkl"
            expr = physo.read_pareto_pkl(str(expression_path))
            expr = str(expr.get_infix_sympy(evaluate_consts=True)[0])
            expr = simplify_expr_str(expr, ndigits=n_digits)
            best_expr_counter.update([expr])

        title = (
            "Frequency of most rewarded expressions over folds\n"
            f"Eq: {eq_title.replace('_', ' ')}"
        )
        if task == "classification":
            title += (f" - {imbalance}\\% pos. samples")
        save_path = plots_dir / f"{plot_prefix}_best_expression_frequency.png"
        plot_expression_frequency(best_expr_counter, title, save_path)

        # =========================
        # PARETO FRONT EXPRESSIONS
        # =========================
        pareto_counter = Counter()
        for fold in range(n_folds):
            fold_dir = results_dir / f"fold_{fold}"
            pareto_csv_path = (fold_dir / "training_curves_pareto.csv")
            pareto_front_formulas = physo.read_pareto_csv(str(pareto_csv_path))
            pareto_front_formulas = [str(formula) for formula in pareto_front_formulas]
            # Remove unsafe and duplicate expressions
            safe_expressions = get_safe_unique_expressions(pareto_front_formulas)
            pareto_front_expressions = [simplify_expr_str(str(formula), ndigits=n_digits) for formula in safe_expressions]
            pareto_counter.update(pareto_front_expressions)

        title = (
            "Frequency of Pareto front expressions over folds\n"
            f"Eq: {eq_title.replace('_', ' ')}"
        )
        if task == "classification":
            title += (f" - {imbalance}\\% pos. samples")
        save_path = plots_dir / f"{plot_prefix}_pareto_front_frequency.png"
        plot_expression_frequency(pareto_counter, title, save_path)
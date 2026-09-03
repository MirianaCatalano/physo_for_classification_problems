import pandas as pd
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

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
    config_path = project_dir / "configs" / "07B_plot_heatmaps_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    
    # =========================
    # LOG CONFIGURATION
    # =========================
    i_eq, eq_title = validate_eq_to_test(tuple(config["EQ_TO_TEST"]))
    experiment_names = [validate_experiment_name(exp) for exp in config["EXPERIMENT_NAMES"]]
    splits_folder_names = [validate_splits_folder_name(split) for split in config["SPLITS_FOLDER_NAMES"]]
    if len(experiment_names) != len(splits_folder_names):
        raise ValueError("EXPERIMENT_NAMES and SPLITS_FOLDER_NAMES must have the same length.")
    model_list = config["MODEL_LIST"]
    n_digits = config["N_DIGITS"]
    logger.info("Configuration parameters loaded successfully.")
    logger.info(f"Equation to test: {i_eq} - {eq_title}")
    logger.info(f"Experiments: {experiment_names}")
    logger.info(f"Splits folders: {splits_folder_names}")
    logger.info(f"Number of digits for expression simplification: {n_digits}")


    # =========================
    # FOLDERS MANAGEMENT
    # =========================
    
    # Create the results folder if it doesn't exist and save the configuration parameters
    equation_dir = project_dir / "tests" / eq_title.replace(".", "_")
    plots_dir = equation_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Equation directory: {equation_dir}")
    logger.info(f"Plots directory: {plots_dir}")
    
    # =========================
    # LOAD STATISTICS SUMMARIES
    # =========================
    statistics_summaries = []
    for experiment_name, splits_folder_name in zip(experiment_names, splits_folder_names):
        experiment_dir = equation_dir / experiment_name
        splits_dir = experiment_dir / splits_folder_name
        if not splits_dir.exists():
            raise ValueError(f"Splits directory not found: {splits_dir}")
        statistics_path = (splits_dir / "statistics_summary.csv")    
        if not statistics_path.exists():
            raise FileNotFoundError(f"Statistics summary not found: {statistics_path}") 
        statistics_summary = pd.read_csv(statistics_path)
        logger.info(f"Loaded statistics from: {statistics_path}")       
        
        # Keep track of the experiment
        statistics_summary.insert(0, "experiment_name", experiment_name)
        statistics_summary.insert(1, "split_name", splits_folder_name)
        statistics_summaries.append(statistics_summary)
    # Combine all experiments
    statistics_summary = pd.concat(statistics_summaries,ignore_index=True)
    imbalance_sets = [set(df["imbalance"].dropna().unique()) for df in statistics_summaries]
    common_imbalance = set.intersection(*imbalance_sets)
    classification_summary = statistics_summary[statistics_summary["task"] == "classification"].copy()
    classification_summary = classification_summary[classification_summary["imbalance"].isin(common_imbalance)]
    
    # =========================
    # SELECT MODELS
    # =========================
    best_expr_summary = classification_summary[classification_summary["model"] == "best_expr"].copy()
    best_expr_mean = best_expr_summary.pivot(index="imbalance", columns="noise",values="reward_test_mean")
    best_expr_std = best_expr_summary.pivot(index="imbalance",columns="noise",values="reward_test_std")
    # Sort axes
    best_expr_mean = best_expr_mean.sort_index().sort_index(axis=1)
    best_expr_std = best_expr_std.reindex(index=best_expr_mean.index, columns=best_expr_mean.columns)
    for model in model_list:
        model_summary = classification_summary[classification_summary["model"] == model].copy() 
        model_mean = model_summary.pivot(index="imbalance", columns="noise", values="reward_test_mean")
        model_std = model_summary.pivot(index="imbalance", columns="noise", values="reward_test_std")   
        # Sort axes
        model_mean = model_mean.sort_index().sort_index(axis=1)
        model_std = model_std.reindex(index=model_mean.index, columns=model_mean.columns)

        fig, axes = plt.subplots(1, 2,figsize=(14, 6), sharey=True)
        best_expr_annotations = (best_expr_mean.round(n_digits).astype(str) + "\n±\n" + best_expr_std.round(n_digits).astype(str))
        model_annotations = (model_mean.round(n_digits).astype(str)+ "\n±\n" + model_std.round(n_digits).astype(str))
        # Y-axis labels
        y_labels = [
            f"{int(float(label))}\\% pos. sampl."
            for label in best_expr_mean.index
        ]
        best_expr_mean.index.name = None
        best_expr_mean.columns.name = None
        model_mean.index.name = None
        model_mean.columns.name = None
        
        sns.heatmap(best_expr_mean, annot=best_expr_annotations, fmt="", 
                    cmap="viridis", vmin=0, vmax=1, ax=axes[0], cbar=False)

        sns.heatmap(model_mean, annot=model_annotations, fmt="",
                    cmap="viridis", vmin=0, vmax=1, ax=axes[1], cbar=True,
                    cbar_kws={"label": "Reward"})
        plt.setp(axes[0].get_yticklabels(), rotation=0, ha="right")
        axes[0].set_title("$\\Phi$-SO\n(best expression)")
        axes[1].set_title(f"$\\Phi$-SO + {model}\n(Pareto front)")
        axes[0].set_yticklabels(y_labels)
        
        fig.text(0.5, 0.05, "Noise level", ha='center')
        fig.suptitle(f"Imbalance vs Noise level\n {eq_title.replace('_', ' ')}")
        plt.tight_layout()
        save_path = (plots_dir / f"heatmap_best_expr_vs_{model}.png")
        plt.savefig(save_path,dpi=300,bbox_inches="tight")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        logger.info(f"Heatmap saved to: {save_path}")
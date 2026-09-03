import hashlib
import json

def validate_n_samples(n_samples):
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("N_SAMPLES must be an integer.")
    if n_samples <= 0:
        raise ValueError("N_SAMPLES must be greater than 0.")
    return n_samples

def validate_noise_level(noise_level):
    if isinstance(noise_level, bool) or not isinstance(noise_level, (int, float)):
        raise TypeError("NOISE_LEVEL must be a number.")
    if not 0 <= noise_level <= 1:
        raise ValueError("NOISE_LEVEL must be between 0 and 1.")
    return float(noise_level)

def validate_quantile_threshold(quantile_threshold):
    if isinstance(quantile_threshold, bool) or not isinstance(quantile_threshold, (int, float)):
        raise TypeError("QUANTILE_THRESHOLD must be a number.")
    if not 0 <= quantile_threshold <= 1:
        raise ValueError("QUANTILE_THRESHOLD must be between 0 and 1.")
    return float(quantile_threshold)

def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("SEED must be an integer.")
    if seed < 0:
        raise ValueError("SEED must be greater than or equal to 0.")
    return seed

def validate_eq_to_test(eq_to_test):
    if not isinstance(eq_to_test, (list, tuple)) or len(eq_to_test) != 2:
        raise ValueError("EQ_TO_TEST must contain an integer (id of the equation) and a string (name of the equation).")
    eq_index, eq_name = eq_to_test
    if isinstance(eq_index, bool) or not isinstance(eq_index, int):
        raise TypeError("The equation index must be an integer.")
    if eq_index < 0:
        raise ValueError("The equation index must be greater than or equal to 0.")
    if not isinstance(eq_name, str):
        raise TypeError("The equation name must be a string.")
    eq_name = "_".join(eq_name.split())
    if not eq_name:
        raise ValueError("The equation name cannot be empty.")
    return eq_index, eq_name

def validate_experiment_name(experiment_name):
    if not isinstance(experiment_name, str):
        raise TypeError("EXPERIMENT_NAME must be a string.")
    experiment_name = "_".join(experiment_name.split())
    if not experiment_name:
        raise ValueError("EXPERIMENT_NAME cannot be empty.")
    return experiment_name

def validate_splits_folder_name(splits_folder_name):
    if not isinstance(splits_folder_name, str):
        raise TypeError("SPLITS_FOLDER_NAME must be a string.")
    splits_folder_name = "_".join(splits_folder_name.split())
    if not splits_folder_name:
        raise ValueError("SPLITS_FOLDER_NAME cannot be empty.")
    return splits_folder_name

def validate_n_folds(n_folds):
    if isinstance(n_folds, bool) or not isinstance(n_folds, int):
        raise TypeError("N_FOLDS must be an integer.")
    if n_folds <= 0:
        raise ValueError("N_FOLDS must be greater than 0.")
    if n_folds % 5 != 0:
        raise ValueError("N_FOLDS must be a multiple of 5.")
    return n_folds

def validate_task(task, threshold):
    if not isinstance(task, str):
        raise TypeError("TASK must be a string.")
    if not task:
        raise ValueError("TASK cannot be empty.")
    task = task.strip().lower()
    if task not in ["classification", "regression"]:
        raise ValueError("TASK must be either 'classification' or 'regression'.")
    if task == "regression" and threshold is not None:
        print(f"WARNING: Regression task does not require any threshold. It will be set to None")
        threshold = None
    elif task == "classification":
        if not isinstance(threshold, str):
            raise ValueError("Thresholds of TASK must be a string.")
    return task, threshold

def validate_metric(metric):
    if not isinstance(metric, str):
        raise TypeError("METRIC must be a string.")

    metric = metric.strip().lower()
    allowed_metrics = [
        "tss",
        "hss1",
        "hss2",
        "ba",
        "f1_score",
        "prec",
        "rec",
        "spec"
    ]
    if metric not in allowed_metrics:
        raise ValueError(
            f"METRIC must be one of: {allowed_metrics}."
        )
    return metric
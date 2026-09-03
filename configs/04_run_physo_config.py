import physo
import torch
import numpy as np
from utils.analysis_utils import custom_reward


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

# Seed for reproducibility
SEED = 0
# Equation to test (index and name)
EQ_TO_TEST = [41, "Perfect gas pressure"]
# List of experiments to process (name of the folder containing the generated dataset)
EXPERIMENT_NAME = "experiment_2"
# List of splits to process associated to each experiment (name of the folder containing the splits)
SPLITS_FOLDER_NAME = "splits_1"
# Tasks to run ("classification" or "regression")
TASK = "classification"
# For classification, quantile threshold folders to process.
# For regression, None.
THRESHOLD = "thr_90" # "thr_75", "thr_90", # None
# Data type: "clean" for noiseless data, "noisy" for noisy data
DATA_TYPE = "noisy"
# Parallel processing
N_JOBS = 4

# ============================================================
# PHYSO CONFIG
# ============================================================

# Maximum length of the generated expressions
MAX_LENGTH = 100
# Operations to use as tokens during training
OP_NAMES = [
    "mul",
    "add",
    "sub",
    "div",
    "inv",
    "n2",
    "sqrt",
    "neg",
    "sin",
    "cos",
    "n3",
    "cbrt",
    "arcsin",
    "arccos",
    "exp",
    "log"
]
# Fixed constants to use as tokens
FIXED_CONSTS = None
# Units of the fixed constants
FIXED_CONSTS_UNITS = None
# Free constants to use as tokens
FREE_CONSTS_NAMES = None
# Units of the free constants
FREE_CONSTS_UNITS = None
# Number of iterations of the training step
EPOCHS = 100


# ============================================================
# REWARD CONFIG
# ============================================================
reward_config = {

    "classification": {
        "reward_function": custom_reward,
        "zero_out_unphysical": True,
        "zero_out_duplicates": True,
        "keep_lowest_complexity_duplicate": True,
        "parallel_mode": False,
        "n_cpus": 8,
    },

    "regression": {
        "reward_function": physo.physym.reward.SquashedNRMSE,
        "zero_out_unphysical": True,
        "zero_out_duplicates": True,
        "keep_lowest_complexity_duplicate": True,
        "parallel_mode": False,
        "n_cpus": 8,
    },
}


# ============================================================
# LEARNING CONFIG
# ============================================================
BATCH_SIZE = 1000 #int(1e5)
GET_OPTIMIZER = lambda model: torch.optim.Adam(
    model.parameters(),
    lr=0.0025,
)
learning_config = {
    "batch_size": BATCH_SIZE,
    "max_time_step": MAX_LENGTH,
    "n_epochs": int(1e9),
    "gamma_decay": 0.7,
    "entropy_weight": 0.005,
    "risk_factor": 0.05,
#    "rewards_computer": rewards_computer,
    "get_optimizer": GET_OPTIMIZER,
    "observe_units": True,
}

# ============================================================
# FREE CONSTANT OPTIMIZATION CONFIG
# ============================================================
free_const_opti_args = {
    "loss": "MSE",
    "method": "LBFGS",
    "method_args": {
        "n_steps": 30,
        "tol": 1e-8,
        "lbfgs_func_args": {
            "max_iter": 4,
            "line_search_fn": "strong_wolfe",
        },
    },
}


# ============================================================
# PRIORS CONFIG
# ============================================================
priors_config = [
    # Priors without arguments
    ("UniformArityPrior", None),
    ("NoUselessInversePrior", None),
    # Length constraint
    ("HardLengthPrior", {"min_length": 2, "max_length": MAX_LENGTH}),
    # Relationship constraints
    ("RelationshipConstraintPrior",{
        "effectors": ["n2", "sqrt", "cbrt", "n3", "sin", "arcsin", "cos", "arccos", "log", "exp"],
        "relationship": "child",
        "targets": ["sqrt", "n2", "n3", "cbrt", "arcsin", "sin", "arccos", "cos", "exp", "log"],
        "max_nb_violations": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        }
    ),
    # Nested functions
    ("NestedFunctions", {"functions": ["exp"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["sqrt"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["sin"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["cos"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["n2"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["n3"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["cbrt"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["log"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["arcsin"], "max_nesting": 1}),
    ("NestedFunctions", {"functions": ["arccos"], "max_nesting": 1}),
    ("NestedTrigonometryPrior", {"max_nesting": 1}),
    # Physical units
    ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}),
]


# ============================================================
# RNN CELL CONFIG
# ============================================================

cell_config = {
    "hidden_size": 128,
    "n_layers": 1,
    "is_lobotomized": False,
}    
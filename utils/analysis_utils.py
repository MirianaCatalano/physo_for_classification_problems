import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV, OrthogonalMatchingPursuit, RidgeCV
from matplotlib.lines import Line2D
from sympy import sympify, simplify, posify, nsimplify, preorder_traversal, Abs, pi, Float
import sympy as sp
import pandas as pd
import json

import re
from collections import defaultdict

'''
divide: protecetd divide function
'''
def divide(a, b):
    return torch.where(b == 0, torch.nan, a / b)

'''
acc: accuracy function
'''
def acc(TN, FP, FN, TP):
    out = divide(TP+TN, FP+FN+TP+TN)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

'''
rec: recall function
'''
def rec(TN, FP, FN, TP):
    out = divide(TP, FN + TP)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

'''
spec: specificity function
'''
def spec(TN, FP, FN, TP):
    out = divide(TN, FP + TN)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

'''
prec: prcision function
'''
def prec(TN, FP, FN, TP):
    out = divide(TP, FP + TP)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

'''
f1_score: f1_score function
'''
def f1_score(TN, FP, FN, TP):
    out = 2*divide(prec(TN, FP, FN, TP)*rec(TN, FP, FN, TP), prec(TN, FP, FN, TP)+rec(TN, FP, FN, TP))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

'''
tss: true skill statistic function
'''
def tss(TN, FP, FN, TP):
    return rec(TN, FP, FN, TP) + spec(TN, FP, FN, TP) - 1.0

'''
ba: balanced accuracy function
'''
def ba(TN, FP, FN, TP):
    return (tss(TN, FP, FN, TP) + 1)/ 2.0

'''
hss1: HSS1 = (TP-FP)/(FN+TP)
'''
hss1 = lambda TN,FP,FN,TP: np.nan_to_num(divide(TP-FP,FN+TP), posinf=0.0, neginf=0.0)

'''
hss2: HSS2 = (2*(TP*TN)-(FP*FN))/((TP+FN)*(FN+TN)+((TP+FP)*(TN+FP)))
'''
hss2 = lambda TN,FP,FN,TP: np.nan_to_num(divide((2.*((TP*TN)-(FP*FN))),((TP+FN)\
                            *(FN+TN))+((TP+FP)*(TN+FP))), posinf=0.0, neginf=0.0)

'''
Y_predicted: function that binarize the predicted regression output gived a threshold
'''
def Y_predicted(p, tau_star):
    return (p > tau_star).long()

'''
optimal_threshold: function that compute the best threshold maximizing the ba value
'''
def optimal_threshold(Y_train, p, N_tau=100):
    # Y_train: tensor shape (N,)
    # p:       tensor shape (N,)

    a = torch.min(p)
    b = torch.max(p)
    delta = (b - a) / N_tau

    best_BA = torch.tensor(-1.0)
    best_tau = torch.tensor(0.0)

    for j in range(N_tau):
        tau = a + j * delta

        # Predicted binary labels
        Y_pred = Y_predicted(p, tau)

        # Compute confusion matrix values
        TP = torch.sum((Y_train == 1) & (Y_pred == 1)).float()
        FN = torch.sum((Y_train == 1) & (Y_pred == 0)).float()
        FP = torch.sum((Y_train == 0) & (Y_pred == 1)).float()
        TN = torch.sum((Y_train == 0) & (Y_pred == 0)).float()

        BA = ba(TN, FP, FN, TP)
        if BA > best_BA:
            best_BA = BA
            best_tau = tau
            
    return best_tau, best_BA

'''
custom_reward: reward function
'''    
def custom_reward(y_target: torch.Tensor, y_pred: torch.Tensor, y_weights=None):
    # Cast to 1D
    y_true = y_target.reshape(-1).float()
    p      = y_pred.reshape(-1).float()
    # Binary reward based on BA optimal thresholding
    _, reward = optimal_threshold(y_true, p, N_tau=100) 
    return reward

'''
evaluate_formulas_on_dataset: programs computed over the dataset X
'''
def evaluate_formulas_on_dataset(programs, X):
    """
    Evaluate Pareto-front expressions on dataset X.

    Parameters
    ----------
    programs : iterable
        Symbolic expressions to evaluate.
    X : torch.Tensor
        Input data with shape (n_features, n_samples).

    Returns
    -------
    np.ndarray
        Symbolic features with shape (n_samples, n_expressions).
    """
    feature_list = []
    for prog in programs:
        y_feat = prog.execute(X).detach().cpu().numpy().ravel()
        feature_list.append(y_feat)
    if not feature_list:
        raise ValueError("No symbolic expressions were provided.")
    return np.stack(feature_list, axis=1) # shape: (n_samples, n_features)


def group_forms(columns):
    groups = defaultdict(list)
    columns_set = set(columns)
    for col in columns:
        # Numeric constants such as "0.123"
        if re.fullmatch(r"0\.\d+", col):
            base = "0"
        else:
            parts = col.rsplit(".", 1)
            if len(parts) == 2 and parts[1].isdigit():
                candidate_base = parts[0]
                # Consider it a duplicate only if the base exists
                if candidate_base in columns_set:
                    base = candidate_base
                else:
                    base = col
            else:
                base = col
        groups[base].append(col)
    return groups

def get_safe_unique_expressions(expressions):
    """
    Identify safe and unique symbolic expressions.

    Symbolic expressions are grouped according to their base expression.
    Unsafe groups are completely removed, while duplicate versions of a
    safe expression are reduced to a single expression.

    Parameters
    ----------
    expressions : list-like
        Symbolic expressions, typically the columns of a Pareto-front
        symbolic dataset.

    Returns
    -------
    safe_expressions : list
        Expressions to keep after removing unsafe and duplicate expressions.
    """

    unsafe_keywords = ["zoo", "nan", "inf"]
    groups = group_forms(expressions)

    safe_expressions = []
    for base, cols in groups.items():
        base_lower = base.lower()
        is_unsafe_keyword = (
            any(kw in base_lower for kw in unsafe_keywords)
            or "I" in base
            or re.search(r"(?<![A-Za-z])E(?![A-Za-z])", base) is not None
        )
        is_zero_constant = base.strip() == "0"
        has_div_zero = re.search(r"/\s*0(?!\.\d)", base) is not None
        has_pow_zero = re.search(r'\*\*\s*0(?!\.\d)', base) is not None
        has_mul_zero = re.search(r'\*\s*0(?!\.\d)', base) is not None

        is_unsafe = (is_unsafe_keyword or is_zero_constant or has_div_zero or has_pow_zero or has_mul_zero)
        if is_unsafe:
            print(f"Removing unsafe symbolic feature group: '{base}'")
            continue
        else:
            # Keep only the first version of duplicate expressions
            if len(cols) > 1:
                print(f"Removing duplicate symbolic features: {cols[1:]}")
            # Keep only the first expression of each duplicate group
            safe_expressions.append(cols[0])
    return safe_expressions


'''
initialize_model: function that inizialize the model given a fixed seed
'''    
def initialize_model(model_name, seed, task):
    if model_name == "LassoCV":
        model = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, precompute='auto',
                      max_iter=1000000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False,
                      random_state=seed, selection='cyclic')
    elif model_name == "ElasticNetCV":
        model = ElasticNetCV(l1_ratio=[.5, .8], eps=0.001, n_alphas=100,
                             alphas=None, fit_intercept=True, precompute='auto', max_iter=1000000,
                             tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1,
                             positive=False, random_state=seed, selection='cyclic')
    elif model_name == "OrthogonalMatchingPursuit":
        model =  OrthogonalMatchingPursuit(n_nonzero_coefs=None, tol=0.0001, fit_intercept=True, precompute='auto')
    elif model_name == "RidgeCV":
        if task == "classification":
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        if task == "regression":
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        model = RidgeCV(alphas=np.logspace(-4, 4, 100), fit_intercept=True, scoring=None,cv=cv)
    else:
        raise ValueError("Model not recognized. Choose from: LassoCV, RidgeCV, ElasticNetCV, OrthogonalMatchingPursuit")
    return model






# ============
# LOAD RESULTS
# ============
def load_results(results_dir, n_folds, filename):
    rewards = []
    for fold in range(n_folds):
        fold_dir = results_dir / f"fold_{fold}"
        results_path = fold_dir / filename
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        df = pd.read_csv(results_path)
        df.insert(0, "fold", fold)
        # Best expression: one row, no model column
        if "model" not in df.columns:
            df.insert(1, "model", "best_expr")
        rewards.append(df)
    return pd.concat(rewards, ignore_index=True)


# =====================================
# COMPUTE CLASSIFICATION METRICS
# =====================================
def compute_metrics(confusion_matrices):
    metrics = confusion_matrices[["fold", "model", "noise", 
                                  "task", "imbalance", 
                                  "threshold", "data_type"]].copy()
    for dataset in ["train", "test"]:
        for i in range(len(confusion_matrices)):
            tn = torch.tensor(confusion_matrices[f"TN_{dataset}"].iloc[i], dtype=torch.float32)
            fp = torch.tensor(confusion_matrices[f"FP_{dataset}"].iloc[i], dtype=torch.float32)
            fn = torch.tensor(confusion_matrices[f"FN_{dataset}"].iloc[i], dtype=torch.float32)
            tp = torch.tensor(confusion_matrices[f"TP_{dataset}"].iloc[i], dtype=torch.float32)
            
            metrics.loc[i, f"BA_{dataset}"] = ba(tn, fp, fn, tp).numpy()
            metrics.loc[i, f"ACC_{dataset}"] = acc(tn, fp, fn, tp).numpy()
            metrics.loc[i, f"HSS1_{dataset}"] = hss1(tn, fp, fn, tp)
            metrics.loc[i, f"HSS2_{dataset}"] = hss2(tn, fp, fn, tp)
            metrics.loc[i, f"REC_{dataset}"] = rec(tn, fp, fn, tp).numpy()
            metrics.loc[i, f"SPEC_{dataset}"] = spec(tn, fp, fn, tp).numpy()
            metrics.loc[i, f"PREC_{dataset}"] = prec(tn, fp, fn, tp).numpy()
            metrics.loc[i, f"F1_SCORE_{dataset}"] = f1_score(tn, fp, fn, tp).numpy()
    return metrics



def find_available_results(experiment_dir, splits_dir, data_type):
    """
    Find all available results for the selected experiment, split configuration and data type.
    """

    # Read noise level from dataset_config.json
    dataset_config_path = experiment_dir / "generated_dataset" / "dataset_config.json"
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)
    noise_level = dataset_config["noise_level"]
    
    results = []
    
    # Look for regression result folder
    results_dir = splits_dir / "results_regression"
    if results_dir.exists():
        results.append({
            "noise": noise_level,
            "task": "regression",
            "imbalance": None,
            "threshold": None,
            "data_type": data_type,
            "results_dir": results_dir
        })
    
    # Look for classification result folders
    for results_dir in splits_dir.glob("results_classification_thr_*"):
        # Extract threshold from folder name and convert to imbalance
        threshold_value = results_dir.name.replace("results_classification_thr_", "")
        imbalance = 100 - float(threshold_value)

        results.append({
            "noise": noise_level,
            "task": "classification",
            "imbalance": imbalance,
            "threshold": float(threshold_value),
            "data_type": data_type,
            "results_dir": results_dir
        })
    return pd.DataFrame(results)



def create_results_summary(rewards_best_expr, metrics_best_expr, rewards_pareto, metrics_pareto):
    """
    Create a single summary DataFrame containing rewards and metrics
    for the best expression and Pareto-front models.
    """

    # ============================
    # BEST EXPRESSION
    # ============================
    summary_best = rewards_best_expr.merge(
        metrics_best_expr,
        on=[
            "fold",
            "model",
            "noise",
            "task",
            "imbalance",
            "threshold",
            "data_type"
        ],
        how="left"
    )

    # ============================
    # PARETO FRONT
    # ============================

    summary_pareto = rewards_pareto.merge(
        metrics_pareto,
        on=[
            "fold",
            "model",
            "noise",
            "task",
            "imbalance",
            "threshold",
            "data_type"
        ],
        how="left"
    )

    # ============================
    # COMBINE
    # ============================
    results_summary = pd.concat(
        [summary_best, summary_pareto],
        ignore_index=True
    )

    return results_summary


def create_statistics_summary(results_summary):
    """
    Compute mean and standard deviation across folds for each
    experimental configuration and model.
    """

    group_columns = [
        "noise",
        "task",
        "imbalance",
        "threshold",
        "data_type",
        "model"
    ]

    # Numeric result columns to aggregate
    value_columns = [column for column in results_summary.columns
        if column not in group_columns + ["fold"]
        and pd.api.types.is_numeric_dtype(results_summary[column])
    ]

    statistics_summary = (
        results_summary
        .groupby(group_columns, dropna=False)[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex columns
    statistics_summary.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple)
        else column
        for column in statistics_summary.columns
    ]
    return statistics_summary


# =========================
# PLOT UTILITIES
# =========================
def plot_expression_frequency(counter, title, save_path):
    """
    Plot the frequency of expressions.
    """
    if not counter:
        raise ValueError("No expressions available for plotting.")

    labels, counts = zip(*counter.most_common())

    fig_width = max(10, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    label_fontsize = max(7, min(14, 400 / len(labels)))

    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [f"${sp.latex(sp.sympify(expr))}$" for expr in labels],
        rotation=60,
        ha="right",
        fontsize=label_fontsize
    )
    ax.set_yticks(np.arange(0, max(counts) + 1, 1))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Expressions", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title(title, fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(3)
    plt.close()





'''
binarize_true_label: function that takes as input the true labels Y 
                    and the class above which to binarize (C or M) 
                    and returns the binarized labels Y_bin
'''
def binarize_true_label(Y, above_class):
    if above_class == 'C':
        Y_bin = ((Y == 'C') | (Y == 'M') | (Y == 'X')).astype(int)
    elif above_class == 'M':
        Y_bin = ((Y == 'M') | (Y == 'X')).astype(int)
    else:
        raise ValueError("Insert only C or M class")
    return Y_bin



def rationalize_exponents_only(expr, max_denominator=12):
    """
    Converte SOLO gli esponenti float (es. 0.5, 0.3333...) in frazioni esatte
    (es. 1/2, 1/3), lasciando invariati tutti gli altri numeri
    (coefficienti fisici, pi, costanti, ecc.).
    """
    replacements = {}
    for node in preorder_traversal(expr):
        if node.is_Pow and node.exp.is_Float:
            candidate = nsimplify(node.exp, rational=True)
            if candidate.is_Rational and candidate.q <= max_denominator:
                replacements[node.exp] = candidate
    return expr.xreplace(replacements)


def apply_protected_root(expr):
    """
    Protegge da radici complesse qualunque potenza x**e con e numerico
    non intero (sia Rational esatto che Float non ancora razionalizzato).
    Per denominatori dispari noti (Rational con q dispari) preserva il segno
    della radice reale; in tutti gli altri casi (denominatore pari, o
    esponente ancora Float e quindi di segno "sconosciuto" a priori) usa
    Abs(base) per garantire un risultato reale.
    """
    replacements = {}
    for node in preorder_traversal(expr):
        if not node.is_Pow:
            continue
        exp = node.exp
        is_noninteger_numeric = (exp.is_Rational and not exp.is_Integer) or \
                                 (isinstance(exp, sp.Float) and exp != int(exp))
        if not is_noninteger_numeric:
            continue

        base = node.base
        coeff, _ = base.as_coeff_Mul()

        if exp.is_Rational:
            p, q = exp.p, exp.q
            if q % 2 != 0 and coeff.is_negative and p % 2 != 0:
                replacements[node] = -Abs(base) ** exp
            else:
                replacements[node] = Abs(base) ** exp
        else:
            # Esponente ancora Float: non possiamo determinare con certezza
            # la parità p/q, quindi usiamo Abs per garantire un risultato reale
            replacements[node] = Abs(base) ** exp

    return expr.xreplace(replacements)

def recognize_pi_multiples(expr, tol=1e-4, max_denom=12, max_power=4):
    """
    Riconosce coefficienti numerici che sono multipli puramente
    moltiplicativi di una potenza di pi (es. 3.1416 -> pi, 6.2832 -> 2*pi,
    31.006 -> pi**3, 1.0472 -> pi/3), scartando qualunque combinazione
    additiva che nsimplify potrebbe trovare ma che non ha significato fisico.
    """
    replacements = {}
    powers = sorted(
        [n for n in range(-max_power, max_power + 1) if n != 0],
        key=abs   # prova prima pi**1 e pi**-1, poi pi**2 e pi**-2, ecc.
    )

    for node in preorder_traversal(expr):
        if isinstance(node, Float) and node != 0:
            for n in powers:
                pi_pow_val = float(pi) ** n
                ratio = float(node) / pi_pow_val
                candidate_ratio = nsimplify(ratio, rational=True, tolerance=tol)

                if candidate_ratio.is_Rational and candidate_ratio.q <= max_denom:
                    reconstructed = float(candidate_ratio) * pi_pow_val
                    if abs(reconstructed - float(node)) < tol:
                        replacements[node] = candidate_ratio * pi**n
                        break   # trovato il miglior n, non serve provare potenze più alte

    return expr.xreplace(replacements)

def clean_integer_floats(expr):
    """
    Converte qualunque Float numericamente intero (es. 1.0, 2.0, -1.0, 0.0)
    nell'oggetto Integer di sympy, sia che compaia come esponente
    (permettendo x**1 -> x) sia come coefficiente moltiplicativo
    (permettendo 1*x -> x), perché sympy applica l'auto-semplificazione
    solo su Integer, non su Float anche se numericamente identici.
    """
    replacements = {}
    for node in preorder_traversal(expr):
        if isinstance(node, sp.Float):
            val = float(node)
            if val == round(val):
                replacements[node] = sp.Integer(round(val))
    return expr.xreplace(replacements)

def round_floats(expr, ndigits=4):
    """
    Arrotonda i coefficienti Float a ndigits decimali per la visualizzazione.
    Se l'arrotondamento farebbe collassare a zero un coefficiente che in
    realtà non è zero, mantiene invece 'ndigits' cifre SIGNIFICATIVE
    (non decimali) per evitare che l'intera espressione collassi a 0
    per un effetto puramente cosmetico dell'arrotondamento.
    """
    def _round(x):
        val = float(x)
        rounded = round(val, ndigits)
        if rounded == 0.0 and val != 0.0:
            # Arrotondamento a cifre significative invece che decimali fisse
            from math import floor, log10
            sig_digits = max(ndigits, 2)
            magnitude = floor(log10(abs(val)))
            decimals_needed = sig_digits - magnitude - 1
            rounded = round(val, decimals_needed)
        return sp.Float(rounded)

    return expr.replace(
        lambda x: isinstance(x, sp.Float),
        _round
    )

    
'''
simplify_expr_str: function that simplify a string expression assuming all symbols are positive
'''   
def simplify_expr_str(expr_str, ndigits = 4):
    expr = sympify(expr_str)
    if expr.has(sp.I):
        print("   -> I già presente qui! Il problema è a monte, nella stringa grezza di physo.")
    
    # Solo gli esponenti diventano frazioni esatte (es. 0.5 -> 1/2)
    expr = rationalize_exponents_only(expr)
    if expr.has(sp.I):
        print("   -> I presente dopo rationalize_exponents_only.")
    
    # Modulo di radici positive (es. protected_sqrt = sqrt(|x|)) per qualunque radice di indice pari
    expr = apply_protected_root(expr) 
    if expr.has(sp.I):
        print("   -> I già presente dopo apply_protected_root.")
    
    # Assume temporaneamente i simboli positivi (grandezze fisiche)
    expr_pos, replacements = posify(expr)
    # simplify prova comunque a riconoscere pi, e, ecc. nei coefficienti,
    # perché qui NON stiamo forzando rational=True sui coefficienti
    simplified = simplify(expr_pos)
    if simplified.has(sp.I):
        print("   -> I già presente dopo simplify.")
    simplified = simplified.subs(replacements)
    
    simplified = recognize_pi_multiples(simplified)
    if simplified.has(sp.I):
        print("   -> I già presente dopo recognize_pi_multiples.")
    # Arrotonda tutti i coefficienti float a 4 decimali
    simplified = round_floats(simplified, ndigits=ndigits)
    if simplified.has(sp.I):
        print("   -> I già presente dopo round_floats.")
    
    simplified = clean_integer_floats(simplified)
    if simplified.has(sp.I):
        print("   -> I già presente dopo clean_integer_floats.")
    return simplified   # ritorna l'oggetto sympy (non str!), utile sia per str() interno che per sp.latex() dopo

'''
correlation_analysis: function that reduce the features to use looking at the
                      ones correlated with each other less than a threshold
'''
def correlation_analysis(correlation_matrix, threshold):
    matrix = correlation_matrix.copy()
    for feat in matrix.columns:
        if feat not in correlation_matrix.columns:
            continue
        this_correlations = correlation_matrix[feat]
        mask = (this_correlations <= threshold)
        this_correlated_features = mask[~mask].index.tolist()
        if len(this_correlated_features) > 1:
            candidate_uncorrelation_score = {
                f: np.linalg.norm(correlation_matrix[f].drop(f))
                for f in this_correlated_features
            }
            min_uncorrelation_score = min(candidate_uncorrelation_score.values()) #min(candidate_scores, key=candidate_scores.get)
            most_uncorrelated_feats = [k for k, v in candidate_uncorrelation_score.items() 
                                    if v == min_uncorrelation_score]
            to_remove = [f for f in this_correlated_features if f not in most_uncorrelated_feats]
            correlation_matrix.drop(columns=to_remove, index=to_remove, inplace=True)
        else:
            continue
    return correlation_matrix

'''
scale_by_order_of_magnitude: rescaling function that reduce the order of
                             magnitude of each feature of the dataset
'''
def scale_by_order_of_magnitude(df, feature_names, eps=1e-30):
    df_scaled = df.copy()
    scales = {}

    for col in feature_names:
        x = df[col].to_numpy()
        x_abs = np.abs(x)
        # Typical value for the feature, computed as the median of the non-zero absolute values
        x_typ = np.median(x_abs[x_abs > eps])
        if x_typ <= eps or np.isnan(x_typ):
            k = 0
        else:
            #k = int(np.floor(np.log10(x_typ)))
            k = int(np.floor(np.log10(x_typ))) + 1  # Adding 1 to ensure that the scaled values are typically around 1 instead of 10
        scale = 10.0 ** (-k)
        df_scaled[col] = x * scale
        scales[col] = scale
    return df_scaled, scales

    


"""
plot_pca: plot function that shows Nox, C and M+ class distribution with a 2D PCA.
"""
def plot_pca(X, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    y = np.asarray(y).ravel()
    label_map = {0: "O", 1: "1"}
    #color_map = {0: "blue", 1: "red"}

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    #classes = np.unique(y)

    '''
    for c in classes:
        pts = X_pca[y == c]  # seleziona solo i punti della classe c
        ax.scatter(pts[:,0], pts[:,1], 
                   #label=label_map.get(c, f"Class {c}"),
                   color=color_map.get(c, "black"),
                   s=10)
    '''
    # Scatter unico (evita che una classe copra l'altra)
    ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=y,
        cmap="bwr",
        s=5,
        vmin=0,
        vmax=1,
        alpha=0.5,
        #edgecolor="#d3d3d3"
    )

    ax.set_box_aspect(1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D")
    ax.grid(False)

    #  --- Ritorna handles e labels "dummy" per legenda globale ---
    '''
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color_map[c], markersize=8, label=label_map[c])
               for c in classes]
    labels = [label_map[c] for c in classes]
    '''
    # --- handles per legenda ---
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='blue', markersize=8, label="O"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='red', markersize=8, label="1")
    ]
    labels = ["O", "1"]

    return ax, handles, labels
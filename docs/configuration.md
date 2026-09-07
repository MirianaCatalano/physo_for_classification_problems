## Configuration

The parameters used throughout the pipeline are specified in the configuration files stored in the `configs/` directory.

Each pipeline step has a dedicated configuration file:

```text
configs/
├── 01_generate_dataset_config.json
├── 02_binarize_label_config.json
├── 03_split_dataset_config.json
├── 04_run_physo_config.py
├── 05_evaluate_expressions_config.json
├── 06_results_summary_config.json
├── 07A_plot_histograms_config.json
└── 07B_plot_heatmaps_config.json
```

The following sections describe the parameters defined in each configuration file.

### `01_generate_dataset_config.json`

This configuration controls the generation of the synthetic datasets from the selected Feynman equation.

| Parameter         | Type             | Description                                                                            |
| ----------------- | ---------------- | -------------------------------------------------------------------------------------- |
| `N_SAMPLES`       | `int`            | Number of samples generated for the dataset.                                           |
| `NOISE_LEVEL`     | `float`          | Noise level used to generate the noisy dataset.                                        |
| `SEED`            | `int`            | Random seed used to ensure reproducibility.                                            |
| `UNIT_NAMES`      | `list[str]`      | Names of the physical units used to represent the variables.                           |
| `EQ_TO_TEST`      | `list[int, str]` | Identifies the Feynman equation to process through its index and name.                 |
| `EXPERIMENT_NAME` | `str`            | Name of the experiment. It is used to identify the corresponding experiment directory. |

### `02_binarize_label_config.json`

This configuration controls the conversion of the continuous regression target into binary classification labels.

| Parameter            | Type             | Description                                                                                                      |
| -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name.                                           |
| `QUANTILE_THRESHOLD` | `list[float]`    | Quantiles used to define the classification thresholds. Each value corresponds to a different binarized dataset. |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment containing the generated dataset to process.                                              |

### `03_split_dataset_config.json`

This configuration controls the generation of the training/test splits.

| Parameter            | Type             | Description                                                            |
| -------------------- | ---------------- | ---------------------------------------------------------------------- |
| `SEED`               | `int`            | Random seed used for reproducible data splitting.                      |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name. |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment containing the dataset to split.                |
| `N_FOLDS`            | `int`            | Number of folds used when generating the cross-validation splits.      |
| `SPLITS_FOLDER_NAME` | `str`            | Name of the directory in which the generated splits are stored.        |

### `04_run_physo_config.py`

This configuration controls the execution of PhySO and contains both experiment-level and PhySO-specific parameters.

#### Experiment configuration

| Parameter            | Type             | Description                                                                                                              |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `SEED`               | `int`            | Random seed used for reproducibility.                                                                                    |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name.                                                   |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment containing the dataset to process.                                                                |
| `SPLITS_FOLDER_NAME` | `str`            | Name of the directory containing the dataset splits to process.                                                          |
| `TASK`               | `str`            | Task to perform. Can be `"classification"` or `"regression"`.                                                            |
| `THRESHOLD`          | `str` or `None`  | Classification threshold folder to process, such as `"thr_50"`, `"thr_75"`, or `"thr_90"`. Set to `None` for regression. |
| `DATA_TYPE`          | `str`            | Type of dataset to process: `"clean"` for noiseless data or `"noisy"` for noisy data.                                    |
| `N_JOBS`             | `int`            | Number of parallel jobs used when processing the experiments.                                                            |

#### PhySO configuration

| Parameter            | Type             | Description                                                                                       |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| `MAX_LENGTH`         | `int`            | Maximum length allowed for the symbolic expressions generated by PhySO.                           |
| `OP_NAMES`           | `list[str]`      | List of mathematical operations available as tokens during symbolic expression generation.        |
| `FIXED_CONSTS`       | `list` or `None` | Fixed numerical constants available to PhySO. `None` means that no fixed constants are specified. |
| `FIXED_CONSTS_UNITS` | `list` or `None` | Physical units associated with the fixed constants.                                               |
| `FREE_CONSTS_NAMES`  | `list` or `None` | Names of the free constants available to PhySO.                                                   |
| `FREE_CONSTS_UNITS`  | `list` or `None` | Physical units associated with the free constants.                                                |
| `EPOCHS`             | `int`            | Number of iterations/epochs used during the training process.                                     |

#### Reward configuration

The `reward_config` dictionary defines separate reward settings for classification and regression.

| Parameter                          | Type       | Description                                                                                |
| ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `reward_function`                  | `callable` | Reward function used by PhySO to evaluate candidate expressions.                           |
| `zero_out_unphysical`              | `bool`     | Whether expressions violating the physical-unit constraints receive zero reward.           |
| `zero_out_duplicates`              | `bool`     | Whether duplicate expressions receive zero reward.                                         |
| `keep_lowest_complexity_duplicate` | `bool`     | Whether, among duplicate expressions, only the one with the lowest complexity is retained. |
| `parallel_mode`                    | `bool`     | Enables or disables parallel reward computation.                                           |
| `n_cpus`                           | `int`      | Number of CPUs used for reward computation when parallel processing is enabled.            |

For classification, `reward_function` is set to the custom reward defined in `utils.analysis_utils`.

For regression, `reward_function` is set to PhySO's `SquashedNRMSE`.

#### Learning configuration

| Parameter        | Type       | Description                                                            |
| ---------------- | ---------- | ---------------------------------------------------------------------- |
| `BATCH_SIZE`     | `int`      | Number of samples processed in each training batch.                    |
| `GET_OPTIMIZER`  | `callable` | Function returning the optimizer used to train the model.              |
| `batch_size`     | `int`      | Batch size passed to the PhySO learning configuration.                 |
| `max_time_step`  | `int`      | Maximum expression length considered during the learning process.      |
| `n_epochs`       | `int`      | Maximum number of learning epochs.                                     |
| `gamma_decay`    | `float`    | Discount factor controlling the decay of future rewards.               |
| `entropy_weight` | `float`    | Weight assigned to the entropy term during learning.                   |
| `risk_factor`    | `float`    | Risk factor used by the learning algorithm.                            |
| `get_optimizer`  | `callable` | Optimizer factory used to create the optimizer.                        |
| `observe_units`  | `bool`     | Specifies whether physical units are provided to the learning process. |

The optimizer is Adam with a learning rate of `0.0025`.

#### Free-constant optimization configuration

The `free_const_opti_args` dictionary controls the optimization of free constants appearing in the generated expressions.

| Parameter        | Type    | Description                                                           |
| ---------------- | ------- | --------------------------------------------------------------------- |
| `loss`           | `str`   | Loss function used to optimize the free constants.                    |
| `method`         | `str`   | Optimization method used for the free constants.                      |
| `n_steps`        | `int`   | Number of optimization steps performed by the selected method.        |
| `tol`            | `float` | Tolerance used as the stopping criterion for the optimization.        |
| `max_iter`       | `int`   | Maximum number of iterations used by the underlying L-BFGS optimizer. |
| `line_search_fn` | `str`   | Line-search strategy used by L-BFGS.                                  |

#### Priors configuration

The `priors_config` list defines the constraints imposed on the expressions generated by PhySO.

| Parameter               | Type        | Description                                                         |
| ----------------------- | ----------- | ------------------------------------------------------------------- |
| `UniformArityPrior`     | `None`      | Prior controlling the arity of generated tokens.                    |
| `NoUselessInversePrior` | `None`      | Prevents unnecessary inverse operations.                            |
| `min_length`            | `int`       | Minimum allowed expression length.                                  |
| `max_length`            | `int`       | Maximum allowed expression length.                                  |
| `effectors`             | `list[str]` | Operations whose relationships with their children are constrained. |
| `relationship`          | `str`       | Type of relationship constrained by the prior.                      |
| `targets`               | `list[str]` | Operations targeted by the relationship constraint.                 |
| `max_nb_violations`     | `list[int]` | Maximum number of allowed violations for each target operation.     |
| `functions`             | `list[str]` | Functions to which a nesting constraint is applied.                 |
| `max_nesting`           | `int`       | Maximum allowed nesting depth for the specified functions.          |
| `prob_eps`              | `float`     | Numerical tolerance used by the physical-units prior.               |

The configuration also includes `NestedTrigonometryPrior`, which limits the nesting of trigonometric functions, and `PhysicalUnitsPrior`, which enforces consistency with the physical units of the problem.

#### RNN cell configuration

The `cell_config` dictionary controls the architecture of the recurrent neural network used by PhySO.

| Parameter        | Type   | Description                                                        |
| ---------------- | ------ | ------------------------------------------------------------------ |
| `hidden_size`    | `int`  | Number of hidden units in the RNN cell.                            |
| `n_layers`       | `int`  | Number of recurrent layers.                                        |
| `is_lobotomized` | `bool` | Specifies whether the RNN cell uses the lobotomized configuration. |

### `05_evaluate_expressions_config.json`

This configuration controls the evaluation of the expressions generated by PhySO.

| Parameter            | Type             | Description                                                                                 |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------- |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name.                      |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment containing the results to evaluate.                                  |
| `SPLITS_FOLDER_NAME` | `str`            | Name of the directory containing the dataset splits.                                        |
| `TASK`               | `str`            | Task to evaluate: `"classification"` or `"regression"`.                                     |
| `THRESHOLD`          | `str` or `None`  | Classification threshold folder to evaluate, such as `"thr_50"`, `"thr_75"`, or `"thr_90"`. |
| `DATA_TYPE`          | `str`            | Type of dataset to evaluate: `"clean"` or `"noisy"`.                                        |
| `MODEL_LIST`         | `list[str]`      | List of models whose results are evaluated by the analysis scripts.                         |

### `06_results_summary_config.json`

This configuration controls the aggregation and summarization of the evaluation results.

| Parameter            | Type             | Description                                                            |
| -------------------- | ---------------- | ---------------------------------------------------------------------- |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name. |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment whose results are summarized.                   |
| `SPLITS_FOLDER_NAME` | `str`            | Name of the directory containing the dataset splits and results.       |
| `DATA_TYPE`          | `str`            | Type of dataset whose results are summarized: `"clean"` or `"noisy"`.  |

### `07A_plot_histograms_config.json`

This configuration controls the generation of the result histograms.

| Parameter            | Type             | Description                                                                    |
| -------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `EQ_TO_TEST`         | `list[int, str]` | Identifies the Feynman equation to process through its index and name.         |
| `EXPERIMENT_NAME`    | `str`            | Name of the experiment whose results are visualized.                           |
| `SPLITS_FOLDER_NAME` | `str`            | Name of the directory containing the splits and summarized results.            |
| `DATA_TYPE`          | `str`            | Type of dataset to visualize: `"clean"` or `"noisy"`.                          |
| `N_DIGITS`           | `int`            | Number of decimal digits displayed in the numerical values shown in the plots. |

### `07B_plot_heatmaps_config.json`

This configuration controls the generation of the result heatmaps.

| Parameter             | Type             | Description                                                            |
| --------------------- | ---------------- | ---------------------------------------------------------------------- |
| `EQ_TO_TEST`          | `list[int, str]` | Identifies the Feynman equation to process through its index and name. |
| `EXPERIMENT_NAMES`    | `list[str]`      | List of experiments whose results are included in the heatmaps.        |
| `SPLITS_FOLDER_NAMES` | `list[str]`      | List of split directories corresponding to the experiments.            |
| `DATA_TYPE`           | `str`            | Type of dataset to visualize: `"clean"` or `"noisy"`.                  |
| `N_DIGITS`            | `int`            | Number of decimal digits displayed in the heatmap annotations.         |
| `MODEL_LIST`          | `list[str]`      | List of models included in the heatmap analysis.                       |

Using configuration files separates the experimental parameters from the implementation of the pipeline, allowing the same scripts to be reused with different experimental settings without modifying the source code.

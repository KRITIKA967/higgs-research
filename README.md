# HIGGS Boson Class Imbalance Benchmark

This repository contains a complete, unique, and publication-quality machine learning research project benchmark. It evaluates **5 class imbalance handling techniques** across **7 machine learning models** under **3 imbalance ratios** (a total of **105 experimental configurations**) using the **HIGGS Boson dataset** from the UCI Machine Learning Repository.

Targeted for submission to arXiv and college course presentation, this repository provides self-contained Jupyter notebooks structured sequentially.

---

## ⚠️ CRITICAL RAM WARNING — READ BEFORE RUNNING CODE

The full `HIGGS.csv` dataset contains 11 million rows and requires approximately **8GB of RAM** just to load, which can easily exceed the limits of a standard 16GB RAM machine when training memory-intensive algorithms like Random Forest or CatBoost.

- **`SAMPLE_MODE = True` is the default across all notebooks.** This loads a safe subset of **500,000 rows** (~1.5GB RAM usage).
- **NEVER** change the default `SAMPLE_MODE` setting in source code to `False` unless you are executing the final overnight run.
- Ensure all other heavy applications (e.g., Chrome, IDEs) are closed before attempting a full-dataset run (`SAMPLE_MODE = False`).
- The code prints a clear RAM warning before every data load operation.

---

## Physics Context & Dataset

In particle physics, identifying the Higgs boson against background noise is a classic binary classification challenge:
- **Class 1 (Signal)**: Higgs boson decay events.
- **Class 0 (Background)**: Standard Model background processes.

The raw dataset (`HIGGS.csv`, ~8GB) contains 11M rows and 28 physics-derived features:
- Columns 2–22: Low-level kinematic features.
- Columns 23–29: High-level physics features derived by physicists.

No missing values exist in the dataset.

### Imbalance Ratios (Versions)
We construct three versions of the dataset:
- **Version A (1:1)**: Original balance (approx. 53% signal, 47% background).
- **Version B (1:10)**: Signal downsampled to 1/10th of the background count.
- **Version C (1:50)**: Signal downsampled to 1/50th of the background count (extreme imbalance).

---

## Benchmark Details

### Machine Learning Models
1. **XGBoost**: Extreme Gradient Boosting.
2. **LightGBM**: Fast, histogram-based gradient boosting.
3. **Random Forest**: Bagging-based ensemble of decision trees.
4. **CatBoost**: Categorical gradient boosting.
5. **AdaBoost**: Adaptive boosting (with limitations).
6. **Voting Ensemble (3-Model)**: Soft-voting combination of XGBoost, LightGBM, and Random Forest.
7. **Voting Ensemble (4-Model)**: Soft-voting combination of XGBoost, LightGBM, Random Forest, and CatBoost.

#### AdaBoost Limitations:
- Does **not** support native `scale_pos_weight` parameter. Sample weights are manually computed using a balanced class weight formula and passed during training.
- Does **not** support custom objectives (focal loss). Focal loss runs are skipped and replaced with a `class_weight='balanced'` equivalent.
- Exhibits significantly slower execution times on large datasets compared to modern gradient-boosted trees.
- Constraints are detailed as comments in [11_experiment10_adaboost.ipynb](file:///D:/higgs_Research/notebooks/11_experiment10_adaboost.ipynb).

### Class Imbalance Techniques
1. **Baseline**: No modification, default classification threshold (0.5).
2. **Class Weights**: Penalizes minority class misclassifications heavily.
3. **SMOTE**: Synthetic Minority Over-sampling Technique. Applied exclusively to the training split.
4. **Focal Loss**: A custom loss function that dynamically down-weights easy background examples and forces the model to focus on hard signal examples.
5. **Threshold Optimization**: Decision threshold scanned from 0.1 to 0.9 on validation set, choosing the threshold that maximizes the F1-score.

---

## Folder Structure

```
D:\higgs_Research\
├── HIGGS.csv                              ← already exists (do not touch/move)
├── requirements.txt                       ← project dependencies
├── README.md                              ← this file
├── data\                                  ← saved train/test CSVs
├── figures\                               ← all PNG figures (dpi=300)
├── results\                               ← all CSVs + .npy probability arrays
├── notebooks\
│   ├── 00_dataset_prep.ipynb              ← prepares data splits for versions A, B, C
│   ├── 01_eda.ipynb                       ← exploratory data analysis & distributions
│   ├── 02_experiment1_baseline_xgb.ipynb  ← XGBoost Baseline
│   ├── 03_experiment2_class_weights_xgb.ipynb ← XGBoost Class Weights
│   ├── 04_experiment3_smote_xgb.ipynb     ← XGBoost SMOTE
│   ├── 05_experiment4_focal_loss_xgb.ipynb ← XGBoost Focal Loss + Threshold Opt.
│   ├── 06_experiment5_threshold_xgb.ipynb  ← XGBoost Threshold Optimization
│   ├── 07_experiment6_combined_xgb.ipynb   ← XGBoost Combined
│   ├── 08_experiment7_lightgbm.ipynb      ← LightGBM Experiments (5 techniques)
│   ├── 09_experiment8_random_forest.ipynb ← Random Forest Experiments (5 techniques)
│   ├── 10_experiment9_catboost.ipynb      ← CatBoost Experiments (5 techniques)
│   ├── 11_experiment10_adaboost.ipynb     ← AdaBoost Experiments (5 techniques)
│   ├── 12_experiment11_voting_3model.ipynb ← 3-Model Voting Ensemble
│   ├── 13_experiment12_voting_4model.ipynb ← 4-Model Voting Ensemble
│   ├── 14_compile_results_xgb.ipynb       ← compiles and compares XGBoost runs
│   └── 15_compile_results_v2_all_models.ipynb ← compiles and compares all 7 models
└── utils\
    ├── __init__.py                        ← makes utils a package
    ├── data_loader.py                     ← dataset loaders & splits
    └── metrics.py                         ← metrics: AUC, F1, Signal Significance
```

---

## Execution Guide

To replicate this research project benchmark:

1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Execute Prep and EDA**:
   - Run `00_dataset_prep.ipynb` to construct and save dataset splits.
   - Run `01_eda.ipynb` to verify distributions and create Figure 1 & Figure 7.
3. **Execute Model Notebooks**:
   - Run notebooks `02` through `13` in order to train and evaluate all models. Each notebook writes output metrics to the `results/` folder and saves test probabilities.
4. **Compile Results**:
   - Run `14_compile_results_xgb.ipynb` to generate XGBoost-specific comparative analyses.
   - Run `15_compile_results_v2_all_models.ipynb` to produce comparative tables and the final plots for the paper (ROC curves, F1-scores, AUC, Signal Significance, and Training Times).

# 🔬 HIGGS Boson — Class Imbalance Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-latest-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**The most comprehensive class imbalance benchmark on the HIGGS UCI dataset.**
*7 models × 5 techniques × 3 imbalance ratios = 105 experimental runs.*

[📊 View Results](#-key-results) • [🚀 Quick Start](#-quick-start) • [📓 Notebooks](#-notebooks) • [📄 Paper](#-paper)

</div>

---

## 🌟 Why This Project?

Every second, CERN's Large Hadron Collider produces millions of collision events. Finding a single Higgs boson signal buried in thousands of background events is one of the hardest classification problems in science.

**The challenge:** At 1:50 signal-to-background ratio, a naive model achieves 98% accuracy by predicting *background for everything* — and catches **zero** Higgs bosons.

This project systematically answers: **which technique actually works, on which model, at which imbalance level?**

---

## 🏆 Key Results

> All results on Version C (1:50 imbalance) — the hardest setting.

| Rank | Model | Technique | AUC | F1 | Signal Z | 
|------|-------|-----------|-----|-----|----------|
| 🥇 | Voting 4-Model | Threshold Opt. | **0.7897** | 0.175 | 5.8 |
| 🥈 | CatBoost | Focal Loss | 0.789 | **0.188** | 5.84 |
| 🥉 | XGBoost | Focal Loss | 0.782 | 0.187 | **5.84** |
| 4 | LightGBM | Focal Loss | 0.786 | 0.184 | 5.72 |
| 5 | Voting 3-Model | Focal Loss | 0.784 | 0.189 | ~5.8 |
| 6 | Random Forest | Class Weights | 0.755 | 0.085 | 4.60 |
| 7 | AdaBoost | Any | 0.714 | 0.099 | 4.33 |

> ⚛️ **Physics highlight:** XGBoost + Focal Loss achieves **Z = 5.84** — crossing the Z ≥ 5 particle physics discovery threshold.

---

## 📐 Project Architecture

```
HIGGS Boson (11M rows, 28 features)
         │
         ▼
┌─────────────────────────────────────┐
│         3 Imbalance Versions        │
│  A (1:1)  │  B (1:10)  │  C (1:50) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         5 Imbalance Techniques      │
│  Baseline │ Class Weights │  SMOTE  │
│     Focal Loss  │ Threshold Opt.    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           7 ML Models               │
│ XGBoost │ LightGBM │ Random Forest  │
│ CatBoost │ AdaBoost │ Voting-3 │ V4 │
└─────────────────────────────────────┘
         │
         ▼
  105 Experimental Runs
  AUC + F1 + Signal Significance Z
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.12 (64-bit) | `py -3.12 --version` |
| Git | Any | `git --version` |
| Git LFS | Any | `git lfs version` |

### Installation

```powershell
# 1. Clone the repo
git clone https://github.com/RUDRAIndia/higgs-research.git
cd higgs-research

# 2. Create virtual environment (64-bit Python 3.12)
py -3.12 -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register Jupyter kernel
python -m ipykernel install --user --name=higgs_venv --display-name "higgs_venv"

# 5. Download HIGGS.csv and place in project root
# https://archive.ics.uci.edu/dataset/280/higgs
# (~7.5 GB unzipped)
```

> ⚠️ **RAM Warning:** Full dataset requires ~8GB RAM. Default `SAMPLE_MODE = True` loads 500k rows safely. Flip to `False` only for the final overnight run.

---

## 📓 Notebooks

Run in this exact order:

| # | Notebook | Description | Time (Sample) |
|---|----------|-------------|---------------|
| 00 | `00_dataset_prep.ipynb` | Load HIGGS, create 3 versions, save splits | ~2 min |
| 01 | `01_eda.ipynb` | Class distributions, feature histograms | ~1 min |
| 02 | `02_experiment1_baseline_xgb.ipynb` | XGBoost — no imbalance handling | ~5 min |
| 03 | `03_experiment2_class_weights_xgb.ipynb` | XGBoost + scale_pos_weight | ~5 min |
| 04 | `04_experiment3_smote_xgb.ipynb` | XGBoost + SMOTE oversampling | ~10 min |
| 05 | `05_experiment4_focal_loss_xgb.ipynb` | XGBoost + Focal Loss + Threshold | ~8 min |
| 06 | `06_experiment5_threshold_xgb.ipynb` | XGBoost + Threshold Optimization | ~5 min |
| 07 | `07_experiment6_combined_xgb.ipynb` | XGBoost + Class Weights + Threshold | ~5 min |
| 08 | `08_experiment7_lightgbm.ipynb` | LightGBM × 5 techniques | ~15 min |
| 09 | `09_experiment8_random_forest.ipynb` | Random Forest × 5 techniques | ~20 min |
| 10 | `10_experiment9_catboost.ipynb` | CatBoost × 5 techniques | ~20 min |
| 11 | `11_experiment10_adaboost.ipynb` | AdaBoost × 5 techniques ⚠️ | ~30 min |
| 12 | `12_experiment11_voting_3model.ipynb` | XGB + LGB + RF ensemble | ~25 min |
| 13 | `13_experiment12_voting_4model.ipynb` | XGB + LGB + RF + CAT ensemble | ~30 min |
| 14 | `14_compile_results_xgb.ipynb` | XGBoost figures + tables | ~2 min |
| 15 | `15_compile_results_v2_all_models.ipynb` | Master comparison all models | ~3 min |

**Total sample mode: ~3-4 hours | Full run: ~8-15 hours overnight**

---

## 📊 Evaluation Metrics

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| **AUC** | Area under ROC curve | Standard ML benchmark, threshold-independent |
| **F1-score** | 2·P·R / (P+R) | Directly measures minority class detection |
| **Signal Significance** | Z = S / √(S+B) | Physics standard — Z ≥ 5 = discovery |

> Accuracy is intentionally excluded — at 1:50 imbalance, a model predicting all-background achieves 98% accuracy while catching zero signal events.

---

## 🧪 The 5 Techniques

```
┌──────────────────┬───────────────────────────────────────────────────┐
│ Baseline         │ Standard training. Shows how bad things get.       │
├──────────────────┼───────────────────────────────────────────────────┤
│ Class Weights    │ scale_pos_weight = n_background / n_signal         │
│                  │ No data modification. Fastest technique.           │
├──────────────────┼───────────────────────────────────────────────────┤
│ SMOTE            │ Synthetic minority oversampling on X_train only.   │
│                  │ Version A capped at 500k rows (CPU constraint).    │
├──────────────────┼───────────────────────────────────────────────────┤
│ Focal Loss       │ FL = -(1-p)^γ · log(p), γ=2.0                     │
│                  │ Down-weights easy examples. From RetinaNet (2017). │
├──────────────────┼───────────────────────────────────────────────────┤
│ Threshold Opt.   │ Scan 0.1→0.9 on validation, pick best F1.         │
│                  │ Default 0.5 is wrong for imbalanced data.          │
└──────────────────┴───────────────────────────────────────────────────┘
```

---

## 📁 Output Files

### Figures (`figures/`)

| File | Description | Paper Section |
|------|-------------|---------------|
| `fig1_class_distribution.png` | Class counts across versions | Section 4 |
| `fig2_auc_comparison.png` | AUC — all models at Version C | Section 6 |
| `fig3_f1_comparison.png` | F1 — all models at Version C | Section 6 |
| `fig4_sig_comparison.png` | Signal Z with Z=5 threshold line | Section 6 |
| `fig5_time_comparison.png` | Training time (log scale) | Section 7 |
| `fig6_roc_curves.png` | ROC curves — best configs | Section 6 |
| `fig7_feature_distributions.png` | Feature histograms | Section 4 |

### Results (`results/`)

| File | Description |
|------|-------------|
| `experiment[1-12]_*.csv` | Per-experiment metrics |
| `all_experiments_compiled.csv` | Master table all 105 runs |
| `ranked_best_per_model_version_C.csv` | Best technique per model |

---

## ⚠️ AdaBoost Limitations

AdaBoost is included for benchmark completeness but has known constraints:

- ❌ No `scale_pos_weight` → uses manual `sample_weight` instead
- ❌ No custom objective → focal loss replaced with balanced class weights
- 🐌 Significantly slower than XGBoost/LightGBM on large datasets
- 📉 Worst overall performance (AUC 0.714 at Version C)

These limitations are documented as paper findings, not bugs.

---

## 🔧 Hardware & Environment

| Spec | Value |
|------|-------|
| CPU | Intel i5-10400 @ 2.90GHz |
| RAM | 16 GB |
| GPU | None (CPU only) |
| OS | Windows 11, 64-bit |
| Python | 3.12 (64-bit) |
| Key packages | XGBoost 2.1.1, LightGBM latest, CatBoost latest |

> All experiments reproducible with `random_state=42` everywhere.

---

## 📄 Paper

This project produces all figures and tables for a research paper structured as:

| Section | Content |
|---------|---------|
| Abstract | Problem, dataset, methods, Z=5.84 headline result |
| Introduction | HEP classification, why imbalance matters |
| Related Work | Baldi 2014, SMOTE (Chawla 2002), Focal Loss (Lin 2017) |
| Dataset | HIGGS UCI, 3 imbalance versions |
| Methodology | 5 techniques, 7 models, 3 metrics |
| Results | 15 experimental tables, 7 figures |
| Discussion | Which technique wins and when |
| Conclusion | Answer 3 research questions |

**Headline result for abstract:** *XGBoost + Focal Loss achieves Signal Significance Z = 5.84, crossing the particle physics discovery threshold of Z ≥ 5 at 1:50 imbalance.*

---

## 🤝 Contributing

Found a bug or want to add a new technique/model? PRs are welcome.

1. Fork the repo
2. Create a branch: `git checkout -b feature/new-technique`
3. Add your notebook following the existing naming convention
4. Submit a PR with results CSV included

---

## 📚 References

- Baldi et al. (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning.* Nature Communications.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV (RetinaNet).
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.

---

## ⭐ Star This Repo

If this project helped your research or coursework, please consider giving it a ⭐ — it helps others find it.

---

<div align="center">

Made with ❤️ for physics + machine learning research

**[RUDRAIndia](https://github.com/RUDRAIndia)** 
**[KRITKA967](https://share.google/qARbH3R67XwCYD2s1)** 

</div>
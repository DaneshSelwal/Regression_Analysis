# 🚀 Comprehensive Regression Analysis Pipeline: Advanced Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Adaptive%20CP%20%7C%20HCM%20%7C%20Quantile-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.2-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Welcome to the **End-to-End Regression Analysis Pipeline**. This repository is engineered as a modular, "plug-and-play" framework for robust regression tasks. It goes beyond simple point predictions by integrating a suite of **Uncertainty Quantification (UQ)** methods, ensuring that every prediction is accompanied by a reliable confidence interval.

Whether you are analyzing environmental data, financial time-series, or industrial sensor readings, this pipeline allows you to swap in your dataset and immediately leverage state-of-the-art Hyperparameter Tuning, Quantile Regression, Probabilistic Modeling, Hyperspherical Confidence Mapping, and Adaptive Conformal Prediction.

> **Branch Info:** You are viewing the `version1.2` branch. See [`version-1.1`](https://github.com/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/tree/version-1.1) for the previous release without ACP.

---

## 📑 Table of Contents (Navigation)

1. [🆕 What's New in v1.2](#-whats-new-in-v12)
2. [📌 Project Overview](#-project-overview)
3. [📂 Repository Structure](#-repository-structure)
4. [📊 Dataset & Usage](#-dataset--usage)
5. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 3b: Probabilistic Distribution (CARD)](#phase-3b-probabilistic-distribution-card)
    - [Phase 3c: Hyperspherical Confidence Mapping (HCM)](#phase-3c-hyperspherical-confidence-mapping-hcm)
    - [Phase 4: Standard Conformal Predictions](#phase-4-standard-conformal-predictions)
    - [Phase 5: Adaptive & Non-Exchangeable CP](#phase-5-adaptive--non-exchangeable-cp)
    - [Phase 5b: Adaptive Coverage Policies (ACP)](#phase-5b-adaptive-coverage-policies-acp)
6. [🚀 Getting Started](#-getting-started)

---

## 🆕 What's New in v1.2

This release adds **Phase 5b — Adaptive Coverage Policies (ACP)**, extending the pipeline with a learned, data-dependent approach to conformal coverage:

*   🧠 **AlphaNet Policy Model** — A neural network that learns per-sample miscoverage rates ($\alpha$) from leave-one-out calibration features, replacing static significance levels.
*   📊 **Multi-Model Result Exports** — ACP results exported as `.xlsx` files for 9 regressors (XGBoost, LightGBM, CatBoost, NGBoost, PGBM, GPBoost, Gradient Boosting, HGBM, TabNet).
*   📉 **Training Dynamics Plotting** — Visualisation of loss curves and coverage evolution across training epochs.
*   🔄 **Multi-Lambda Training** — Sweep over regularisation strengths to find the best width-coverage trade-off.

---

## 📌 Project Overview

This framework provides a rigorous path from raw data to confident predictions. It is designed to be **domain-agnostic**: while the inspiration comes from hydrological sediment load analysis, the methods are applicable to any regression problem, especially those involving time-series or non-exchangeable data.

**Key Features:**
*   **Automated Optimization**: Harnessing **Optuna** for Bayesian optimization of complex regressors.
*   **Interval Estimation**: **Quantile Regression** for estimating conditional bounds (e.g., 5th and 95th percentiles).
*   **Full Distribution Modeling**: Using **NGBoost** and **PGBM** to predict the full probability distribution parameters ($\mu, \sigma$).
*   **Generative Modeling**: Leveraging **CARD (Classification and Regression Diffusion)** models to generate conditional distributions using diffusion processes.
*   **Geometric Uncertainty**: **Hyperspherical Confidence Mapping (HCM)** for sampling-free, distribution-free uncertainty estimation in regression.
*   **Robust Uncertainty**: Implementation of **NEXCP (Non-Exchangeable Conformal Prediction)** and **Adaptive CP**, crucial for handling data drift and temporal dependencies where standard methods fail.
*   **Policy-Learned Coverage**: Integration of **Adaptive Coverage Policies (ACP)** to learn data-dependent coverage levels via calibration-aware neural policies.

---

## 📂 Repository Structure

The project is organized into a clean, professional architecture optimized for Google Colab:

```
.
# 📦 Upload these folders to Google Drive
├── Data/               # 📊 Raw Datasets (train.csv, test.csv)
├── HyperParameter_Tuning/
├── Quantile_Regression/
├── Probabilistic_Distribution/
├── Probabilistic_Distribution(CARD)/
├── Hyperspherical_Confidence_Mapping(HCM)/                 # (Added in v1.1)
├── Conformal_Predictions(MAPIE,PUNCC)/
├── Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/
└── conformal_predictions_adaptive_coverage_policies/       # (Added in v1.2)
│
└── README.md               # 🚀 Project Landing Page
```

---

## 🛠️ Workflow & Methodology

**This is a Template Pipeline.**

To use this repository with your own data:

1.  **Prepare your data**: You need a training set and a testing set.
2.  **Format**: Ensure your files are in `.csv` format.
3.  **Replace**:
    *   Place your data in the `Data/` directory.
4.  **Configure**:
    *   **Column Names**: Open the notebooks in `` and ensure the column names match your dataset's target variable and features.
    *   **File Paths**: When running in Google Colab, paths are automatically handled relative to the `Data_folder` root.

---

### Phase 1: Hyperparameter Tuning
**Location**: `HyperParameter_Tuning`
Before any uncertainty quantification, we must ensure our base estimators are accurate.
*   **Tool**: **Optuna**.
*   **Process**: We search over hyperparameter spaces for XGBoost, CatBoost, LightGBM, etc., using efficient pruners (Hyperband) to find the best configuration.
*   **Output**: Optimized model parameters saved for subsequent steps.

### Phase 2: Quantile Regression
**Location**: `Quantile_Regression`
We move beyond the mean.
*   **Goal**: Predict conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) to bracket the target value.
*   **Loss Function**: Pinball Loss.
*   **Result**: A prediction interval that captures a specified percentage of the data (e.g., 90%).

### Phase 3: Probabilistic Distribution
**Location**: `Probabilistic_Distribution`
Treating the target as a random variable $Y|X \sim \mathcal{D}(\theta)$.
*   **Models**: **NGBoost** (Natural Gradient Boosting) and **PGBM** (Probabilistic Gradient Boosting Machines).
*   **Metrics**: Negative Log-Likelihood (NLL) and Continuous Ranked Probability Score (CRPS).
*   **Visualization**: Probability Integral Transform (PIT) histograms to verify calibration.

### Phase 3b: Probabilistic Distribution (CARD)
**Location**: `Probabilistic_Distribution(CARD)`
Using generative diffusion models to capture complex conditional distributions.
*   **Models**: **CARD** (Classification and Regression Diffusion).
*   **Method**: Converts the regression target into a noise distribution and learns to reverse the diffusion process conditioned on features.
*   **Advantage**: Capable of modeling multi-modal distributions and complex dependencies.

### Phase 3c: Hyperspherical Confidence Mapping (HCM)
**Location**: `Hyperspherical_Confidence_Mapping(HCM)`
Using a geometric decomposition to estimate regression uncertainty without sampling or a fixed predictive distribution.
*   **Model**: **HCM** (Hyperspherical Confidence Mapping).
*   **Method**: Decomposes the regression output into a scalar magnitude $R$ and a direction vector $d$, then measures uncertainty through the violation of the unit-norm hyperspherical constraint.
*   **Advantage**: Lightweight, deterministic, and designed to stay compatible with the same Google Colab workflow and Excel-style result exports used across the repository.

### Phase 4: Standard Conformal Predictions
**Location**: `Conformal_Predictions(MAPIE,PUNCC)`
For data that satisfies the **exchangeability** assumption (i.e., order doesn't matter).
*   **Libraries**: `MAPIE`, `PUNCC`.
*   **Methods**: Split Conformal, CV+, Jackknife+.
*   **Guarantee**: Provides marginal coverage guarantees with finite-sample validity.

### Phase 5: Adaptive & Non-Exchangeable CP
**Location**: `Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)`
**Crucial for Time-Series**.
Real-world data often drifts or has temporal dependencies.
*   **NEXCP**: Non-Exchangeable Conformal Prediction. Weights recent observations more heavily to adapt to distribution shifts.
*   **Adaptive CP**: Dynamically updates the interval width $C_t$ based on recent coverage errors.
*   **Result**: Valid coverage even during volatile periods (e.g., market crashes, floods).

### Phase 5b: Adaptive Coverage Policies (ACP)
**Location**: `conformal_predictions_adaptive_coverage_policies`
Learns data-dependent conformal coverage through a neural policy over calibration statistics.
*   **Base Model**: Linear regression backbone for point prediction.
*   **Policy Model**: `AlphaNet` predicts adaptive miscoverage ($\alpha$) from leave-one-out calibration features.
*   **Objective**: Minimize interval width while regularizing the learned coverage policy.
*   **Outputs**: Excel-first reports with training curves, calibration diagnostics, prediction bands, and matrix evaluation.

---

## 🚀 Getting Started (Colab-First)

1.  **Open a notebook in Colab**:
    Click any .ipynb file in this repository and use the **Open in Colab** button.

2.  **Install dependencies from the notebook**:
    Run the first cell to install required packages in Colab (for example: !pip install optuna mapie puncc).

3.  **Upload your dataset to Colab runtime storage**:
    Upload your own 	rain.csv and 	est.csv directly into the active Colab session.

4.  **Update notebook paths for Colab storage**:
    Set file paths to Colab runtime locations (for example: /content/train.csv and /content/test.csv).

5.  **Run notebooks in phase order**:
    Execute notebooks following the **Repository Structure** sequence (Hyperparameter Tuning $\rightarrow$ Quantile/Probabilistic $\rightarrow$ Conformal Predictions).
---

<sub>This repository is a collaborative project developed under the guidance of Dr. Mahesh Pal by Prakriti Bisht and Danesh Selwal.</sub>

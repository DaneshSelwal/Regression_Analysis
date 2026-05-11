# 🚀 Comprehensive Regression Analysis Pipeline: Advanced Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Adaptive%20CP%20%7C%20HCM%20%7C%20Quantile-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.2-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Welcome to the **End-to-End Regression Analysis Pipeline**. This repository is engineered as a modular, "plug-and-play" framework for robust regression tasks. It goes beyond simple point predictions by integrating a suite of **Uncertainty Quantification (UQ)** methods, ensuring that every prediction is accompanied by a reliable confidence interval.

Whether you are analyzing environmental data, financial time-series, or industrial sensor readings, this pipeline allows you to swap in your Dataset and immediately leverage state-of-the-art Hyperparameter Tuning, Quantile Regression, Probabilistic Modeling, Hyperspherical Confidence Mapping, and Adaptive Conformal Prediction.

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
└── Regression/             # 📂 Main Project Folder (Upload this to MyDrive/)
    ├── data/               # 📊 Raw Datasets (train.csv, test.csv)
    ├── hyperparameter_tuning/
    ├── quantile_regression/
    ├── probabilistic_distribution/
    ├── probabilistic_distribution_card/
    ├── hyperspherical_confidence_mapping_hcm/
    ├── conformal_predictions_mapie_puncc/
    ├── conformal_predictions_nexcp_adaptivecp_mfcs/
    ├── conformal_predictions_adaptive_coverage_policies/
    ├── Regression/resources/          # 📚 Research Paper PDFs
    └── examples/           # 📁 Example Datasets
```

---

## 📊 Dataset & Usage

**This is a Template Pipeline.**

To use this repository with your own data:

1.  **Prepare your data**: You need a training set and a testing set.
2.  **Format**: Ensure your files are in `.csv` format.
3.  **Replace**:
    *   Place your data in the `Regression/data/` directory.
4.  **Configure**:
    *   **Column Names**: Open the notebooks in the respective folders and ensure the column names match your Dataset's target variable and features.
    *   **File Paths**: When running in Google Colab, paths are automatically handled relative to the root directory.

---

## 🛠️ Workflow & Methodology

---

### Phase 1: Hyperparameter Tuning
**Location**: `hyperparameter_tuning`
Before any uncertainty quantification, we must ensure our base estimators are accurate.
*   **Tool**: **Optuna**.
*   **Process**: We search over hyperparameter spaces for XGBoost, CatBoost, LightGBM, etc., using efficient pruners (Hyperband) to find the best configuration.
*   **Output**: Optimized model parameters saved for subsequent steps.

### Phase 2: Quantile Regression
**Location**: `quantile_regression`
We move beyond the mean.
*   **Goal**: Predict conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) to bracket the target value.
*   **Loss Function**: Pinball Loss.
*   **Result**: A prediction interval that captures a specified percentage of the data (e.g., 90%).

### Phase 3: Probabilistic Distribution
**Location**: `probabilistic_distribution`
Treating the target as a random variable $Y|X \sim \mathcal{D}(\theta)$.
*   **Models**: **NGBoost** (Natural Gradient Boosting) and **PGBM** (Probabilistic Gradient Boosting Machines).
*   **Metrics**: Negative Log-Likelihood (NLL) and Continuous Ranked Probability Score (CRPS).
*   **Visualization**: Probability Integral Transform (PIT) histograms to verify calibration.

### Phase 3b: Probabilistic Distribution (CARD)
**Location**: `probabilistic_distribution_card`
Using generative diffusion models to capture complex conditional distributions.
*   **Models**: **CARD** (Classification and Regression Diffusion).
*   **Method**: Converts the regression target into a noise distribution and learns to reverse the diffusion process conditioned on features.
*   **Advantage**: Capable of modeling multi-modal distributions and complex dependencies.

### Phase 3c: Hyperspherical Confidence Mapping (HCM)
**Location**: `hyperspherical_confidence_mapping_hcm`
Using a geometric decomposition to estimate regression uncertainty without sampling or a fixed predictive distribution.
*   **Model**: **HCM** (Hyperspherical Confidence Mapping).
*   **Method**: Decomposes the regression output into a scalar magnitude $R$ and a direction vector $d$, then measures uncertainty through the violation of the unit-norm hyperspherical constraint.
*   **Advantage**: Lightweight, deterministic, and designed to stay compatible with the same Google Colab workflow and Excel-style result exports used across the repository.

### Phase 4: Standard Conformal Predictions
**Location**: `conformal_predictions_mapie_puncc`
For data that satisfies the **exchangeability** assumption (i.e., order doesn't matter).
*   **Libraries**: `MAPIE`, `PUNCC`.
*   **Methods**: Split Conformal, CV+, Jackknife+.
*   **Guarantee**: Provides marginal coverage guarantees with finite-sample validity.

### Phase 5: Adaptive & Non-Exchangeable CP
**Location**: `conformal_predictions_nexcp_adaptivecp_mfcs`
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

1.  **Clone/Download the Repository**:
    Download the project and identify the `Regression/` folder.

2.  **Upload to Google Drive**:
    Upload the entire `Regression/` folder directly to your Google Drive root (`MyDrive/`).

3.  **Open a notebook in Colab**:
    Navigate to any `.ipynb` file within your Drive's `Regression/` folder and open it with Google Colab.

4.  **Run the Notebooks**:
    *   The notebooks are pre-configured to mount your drive and look for data in `/content/drive/MyDrive/Regression/data/`.
    *   Execute notebooks in phase order: **Hyperparameter Tuning** $\rightarrow$ **Quantile/Probabilistic** $\rightarrow$ **Conformal Predictions**.

---

## 📚 Resources & References

This project leverages state-of-the-art research in Uncertainty Quantification. Below are the key resources and research papers utilized in this pipeline:

### 📖 Research Papers (PDFs available in `Regression/resources/`)
*   **NEXCP (Non-Exchangeable Conformal Prediction)**: Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). *Conformal prediction beyond exchangeability*. [Local PDF](Regression/resources/NEXCP_beyond_exchangeability.pdf) | [ArXiv](https://arxiv.org/abs/2202.13415)
*   **Adaptive Conformal Prediction**: Gibbs, I., & Candes, E. (2021). *Adaptive conformal inference under distribution shift*. [Local PDF](Regression/resources/Adaptive_Conformal_Inference.pdf) | [ArXiv](https://arxiv.org/abs/2106.01682)
*   **Conformal Prediction Tutorial**: Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. [Local PDF](Regression/resources/Gentle_Intro_to_Conformal_Prediction.pdf) | [ArXiv](https://arxiv.org/abs/2107.07511)

### 🛠️ Libraries & Frameworks
*   **MAPIE**: Model Agnostic Prediction Interval Estimator. [GitHub](https://github.com/scikit-learn-contrib/MAPIE)
*   **PUNCC**: Predictive UNCertainty Calibration and Conformalization. [GitHub](https://github.com/deel-ai/puncc)
*   **Adaptive Coverage Policies (ACP)**: [GitHub](https://github.com/GauthierE/adaptive-coverage-policies)
*   **HCM (Hyperspherical Confidence Mapping)**: [GitHub](https://github.com/Abandoned-Puppy/HCM)
*   **Optuna**: Bayesian Hyperparameter Optimization. [Website](https://optuna.org/)
*   **NGBoost**: Natural Gradient Boosting for Probabilistic Prediction. [Project Page](https://stanfordmlgroup.github.io/projects/ngboost/)
*   **PGBM**: Probabilistic Gradient Boosting Machines. [GitHub](https://github.com/elephaint/pgbm)
*   **Treeffuser / CARD**: Generative Diffusion for Regression. [GitHub](https://github.com/blei-lab/treeffuser)
*   **IBUG**: Instant Bootstrapping for Uncertainty Guidelines. [GitHub](https://github.com/jjbrophy47/ibug)

---

<sub>This repository is a collaborative project developed under the guidance of Dr. Mahesh Pal by Prakriti Bisht and Danesh Selwal.</sub>

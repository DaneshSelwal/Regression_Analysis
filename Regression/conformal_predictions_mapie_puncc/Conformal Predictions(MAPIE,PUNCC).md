# Conformal Prediction for Regression: Theory & Implementation Summary

> **One-line description:** Distribution-free frameworks that construct valid prediction intervals with formal marginal coverage guarantees for regression tasks.

---

## 1. Overview & Intuition

Standard regression models provide point estimates without reliable bounds on their uncertainty. While some models output variance or confidence intervals, these often rely on strict distributional assumptions (e.g., Gaussian noise) that are rarely true in practice. Conformal Prediction (CP) addresses this limitation by wrapping any base regression model to produce prediction intervals that are mathematically guaranteed to contain the true response with a user-specified probability, regardless of the underlying data distribution.

The core insight behind Conformal Prediction is the use of a "nonconformity score," which measures how unusual a sample is compared to previously seen data. By calibrating these scores on a hold-out dataset, the method computes an empirical threshold. For Split Conformal Prediction, this results in constant-width intervals. To handle heteroscedasticity (varying noise levels across the input space), variants like Conformalized Quantile Regression (CQR) calibrate heuristic quantile models, resulting in adaptive intervals that are wider in regions of high uncertainty and tighter in confident regions. 

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X}$ be the input feature space and $\mathcal{Y} = \mathbb{R}$ be the continuous output space. We are given a training set to fit a base model, and a separate calibration set $\mathcal{D}_{\text{cal}} = \{(x_1, y_1), \dots, (x_n, y_n)\}$ of size $n$. Let $\alpha \in (0, 1)$ be the target miscoverage rate, meaning we desire a prediction interval $C(x)$ such that $\mathbb{P}(y \in C(x)) \ge 1 - \alpha$.

### 2.2 Core Definition / Score Function

**Split Conformal Prediction (Absolute Residuals)**

**Equation:**
$$s_i = |y_i - \hat{\mu}(x_i)|$$

**Where:**
- $s_i$ — the nonconformity score for the $i$-th calibration sample
- $y_i$ — the true target value
- $\hat{\mu}(x_i)$ — the point prediction from the base regression model

**What this means:** The score quantifies how much the true label diverges from the model's prediction. Larger scores indicate a worse fit.

**Conformalized Quantile Regression (CQR)**

**Equation:**
$$s_i = \max\{\hat{q}_{\alpha/2}(x_i) - y_i, \; y_i - \hat{q}_{1-\alpha/2}(x_i)\}$$

**Where:**
- $\hat{q}_{\alpha/2}(x_i)$ — the predicted lower conditional quantile
- $\hat{q}_{1-\alpha/2}(x_i)$ — the predicted upper conditional quantile

**What this means:** The score measures the signed distance by which the true target falls outside the heuristic interval formed by the quantile regressors. If the target is inside the interval, the score is negative.

### 2.3 Calibration Threshold

**Equation:**
$$\hat{q} = \text{Quantile}\left( \{s_1, \dots, s_n\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

**Where:**
- $\hat{q}$ — the empirical quantile threshold
- $n$ — the number of calibration samples
- $\alpha$ — the user-specified error rate

**What this means:** We find a score threshold such that a rigorously defined fraction of the calibration data has scores below it. The finite-sample correction $(n+1)$ ensures the mathematical validity of the coverage bound.

### 2.4 Prediction Set Construction

**Equation (Split CP):**
$$C(x_{\text{test}}) = \left[ \hat{\mu}(x_{\text{test}}) - \hat{q}, \; \hat{\mu}(x_{\text{test}}) + \hat{q} \right]$$

**Equation (CQR):**
$$C(x_{\text{test}}) = \left[ \hat{q}_{\alpha/2}(x_{\text{test}}) - \hat{q}, \; \hat{q}_{1-\alpha/2}(x_{\text{test}}) + \hat{q} \right]$$

**What this means:** For a new test point, the interval is constructed by symmetrically expanding the point prediction (or the quantile bounds) by the calibrated threshold $\hat{q}$ to guarantee marginal coverage.

---

## 3. Algorithm

**Input:** Training data, Calibration data $\mathcal{D}_{\text{cal}}$, Test point $x_{\text{test}}$, error rate $\alpha$, Base regression algorithm $\mathcal{A}$.  
**Output:** Prediction interval $C(x_{\text{test}})$.

1. Train the base model on the training set to obtain $\hat{\mu}$ (or $\hat{q}_{\alpha/2}$ and $\hat{q}_{1-\alpha/2}$).
2. For each sample in the calibration set, compute the nonconformity score $s_i$.
3. Sort the scores and extract the $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$-th empirical quantile, denoted $\hat{q}$.
4. Construct the prediction interval for $x_{\text{test}}$ by adjusting the model's output using the computed threshold $\hat{q}$.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Conformal Predictions(MAPIE,PUNCC).ipynb`

### 4.1 Split CP and CV+ via MAPIE
```python
mapie = MapieRegressor(base_estimator, method="plus", cv=KFold(n_splits=5))
mapie.fit(X_train, y_train)
y_pred, int_pred = mapie.predict(X_test, alpha=MISCOVERAGE)
```
**What this does:** Wraps an arbitrary scikit-learn compatible estimator with the MAPIE library to produce conformal intervals. Setting `method="plus"` executes the CV+ variant, which uses cross-validation folds for both model training and score calibration to improve stability.
**Why:** MAPIE abstracts away the manual computation of residuals and empirical quantiles, providing a unified API for standard symmetric conformal regression while maximizing data efficiency through cross-validation.

### 4.2 Conformalized Quantile Regression via PUNCC
```python
upper_quantile_model = model_class(**model_params)
lower_quantile_model = model_class(**model_params)

# Wrap models in a dual predictor
dualpredictor = DualPredictor(
    [lower_quantile_model, upper_quantile_model], is_trained=[True, True]
)

# Initialize the CQR conformal predictor
cqr = CQR(dualpredictor, train=False)
cqr.fit(X_calib=X_calib, y_calib=y_calib)
y_pred_cqr, y_pred_lower_cqr, y_pred_upper_cqr = cqr.predict(X_test, alpha=alpha)
```
**What this does:** Trains separate models for the lower and upper bounds, bundles them into a `DualPredictor`, and calibrates them on a hold-out set using the `CQR` class from the PUNCC library.
**Why:** Standard Split CP produces constant-width intervals for all inputs. CQR uses quantile models to propose instance-dependent interval widths, which are then corrected by the conformal framework to ensure rigorous theoretical coverage guarantees.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

**Design rules for the toy dataset:**
- $n = 5$ calibration samples.
- Target miscoverage $\alpha = 0.20$ (Coverage = 80%).
- 3 test samples designed to expose method differences:
  - **Easy sample:** The model is highly accurate.
  - **Borderline sample:** The model error is exactly near the calibrated threshold limit.
  - **Ambiguous sample:** The model error is unexpectedly massive.

**Calibration Data Probabilities & Predictions:**
| Sample | True $y_i$ | $\hat{\mu}(x_i)$ | $\hat{q}_{\text{low}}(x_i)$ | $\hat{q}_{\text{high}}(x_i)$ |
|--------|------------|------------------|-----------------------------|------------------------------|
| 1      | 10.5       | 10.0             | 9.5                         | 10.5                         |
| 2      | 15.2       | 14.5             | 14.0                        | 15.0                         |
| 3      | 8.0        | 9.0              | 8.5                         | 9.5                          |
| 4      | 20.1       | 20.0             | 19.5                        | 20.5                         |
| 5      | 12.8       | 11.5             | 10.0                        | 13.0                         |

**Test Data Base Predictions:**
| Sample | Type       | True $y$ | $\hat{\mu}(x)$ | $\hat{q}_{\text{low}}(x)$ | $\hat{q}_{\text{high}}(x)$ |
|--------|------------|----------|----------------|---------------------------|----------------------------|
| T1     | Easy       | 10.1     | 10.0           | 9.5                       | 10.5                       |
| T2     | Borderline | 15.0     | 14.0           | 14.2                      | 14.8                       |
| T3     | Ambiguous  | 25.0     | 20.0           | 19.0                      | 21.0                       |

---

### 5.1 Method 1: Split Conformal Prediction

#### Step A — Compute nonconformity scores on the calibration set
Score: $s_i = |y_i - \hat{\mu}(x_i)|$

| Sample Index | True $y_i$ | $\hat{\mu}(x_i)$ | $s_i$ |
|--------------|------------|------------------|-------|
| 1            | 10.5       | 10.0             | 0.5   |
| 2            | 15.2       | 14.5             | 0.7   |
| 3            | 8.0        | 9.0              | 1.0   |
| 4            | 20.1       | 20.0             | 0.1   |
| 5            | 12.8       | 11.5             | 1.3   |

All scores sorted: `[0.1, 0.5, 0.7, 1.0, 1.3]`

#### Step B — Compute the calibration threshold
$$q_\text{level} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n}$$
$$q_\text{level} = \frac{\lceil (5+1)(1-0.20) \rceil}{5} = \frac{\lceil 6 \times 0.80 \rceil}{5} = \frac{\lceil 4.8 \rceil}{5} = \frac{5}{5} = 1.0$$
Because the level clips to 1.0, the threshold equals the maximum score from the calibration set.
$\hat{q} = 1.3$

#### Step C — Build prediction sets for each test sample
Prediction Interval: $[\hat{\mu}(x) - 1.3, \; \hat{\mu}(x) + 1.3]$

| Sample | True $y$ | Threshold | Interval | In set? |
|--------|----------|-----------|----------|---------|
| T1     | 10.1     | 1.3       | [8.7, 11.3]| ✓       |
| T2     | 15.0     | 1.3       | [12.7, 15.3]| ✓       |
| T3     | 25.0     | 1.3       | [18.7, 21.3]| ✗       |

#### Step D — Summary table for this method
| Sample | Prediction Set | Set Size (Width) | Coverage |
|--------|----------------|------------------|----------|
| T1     | [8.7, 11.3]    | 2.6              | ✓        |
| T2     | [12.7, 15.3]   | 2.6              | ✓        |
| T3     | [18.7, 21.3]   | 2.6              | ✗        |

---

### 5.2 Method 2: Conformalized Quantile Regression (CQR)

#### Step A — Compute nonconformity scores on the calibration set
Score: $s_i = \max\{\hat{q}_{\text{low}}(x_i) - y_i, \; y_i - \hat{q}_{\text{high}}(x_i)\}$

| Sample Index | True $y_i$ | $\hat{q}_{\text{low}}$ | $\hat{q}_{\text{high}}$ | $s_i$ |
|--------------|------------|------------------------|-------------------------|-------|
| 1            | 10.5       | 9.5                    | 10.5                    | 0.0   |
| 2            | 15.2       | 14.0                   | 15.0                    | 0.2   |
| 3            | 8.0        | 8.5                    | 9.5                     | 0.5   |
| 4            | 20.1       | 19.5                   | 20.5                    | 0.0   |
| 5            | 12.8       | 10.0                   | 13.0                    | -0.2  |

All scores sorted: `[-0.2, 0.0, 0.0, 0.2, 0.5]`

#### Step B — Compute the calibration threshold
$q_\text{level} = 1.0$ (same logic as above).
We pick the maximum score.
$\hat{q} = 0.5$

#### Step C — Build prediction sets for each test sample
Prediction Interval: $[\hat{q}_{\text{low}}(x) - 0.5, \; \hat{q}_{\text{high}}(x) + 0.5]$

| Sample | True $y$ | Threshold | Interval | In set? |
|--------|----------|-----------|----------|---------|
| T1     | 10.1     | 0.5       | [9.0, 11.0] | ✓       |
| T2     | 15.0     | 0.5       | [13.7, 15.3]| ✓       |
| T3     | 25.0     | 0.5       | [18.5, 21.5]| ✗       |

#### Step D — Summary table for this method
| Sample | Prediction Set | Set Size (Width) | Coverage |
|--------|----------------|------------------|----------|
| T1     | [9.0, 11.0]    | 2.0              | ✓        |
| T2     | [13.7, 15.3]   | 1.6              | ✓        |
| T3     | [18.5, 21.5]   | 3.0              | ✗        |

---

### 5.3 Cross-Method Comparison

| Sample | Split CP       | CQR            |
|--------|----------------|----------------|
| T1     | [8.7, 11.3] ✓  | [9.0, 11.0] ✓  |
| T2     | [12.7, 15.3] ✓ | [13.7, 15.3] ✓ |
| T3     | [18.7, 21.3] ✗ | [18.5, 21.5] ✗ |

**Why the methods differ:**
Split CP enforces a fixed-width interval for all samples regardless of the input's local variance. The width is always strictly 2.6.
CQR utilizes quantile regression to output adaptive interval widths. For sample T2, CQR produces a tighter interval (width 1.6) because the base model predicted a lower variance in that region. For sample T3, CQR produces a wider interval (width 3.0) capturing the higher uncertainty. Although both methods missed the extremely ambiguous T3, CQR uniquely scaled its interval size dynamically while maintaining marginal guarantees.

---

## 6. References

[1] Vovk, V., Gammerman, A., & Shafer, G. "Algorithmic Learning in a Random World." Springer, 2005. [Link](https://link.springer.com/book/10.1007/b138466)  
[2] Romano, Y., Patterson, E., & Candès, E. "Conformalized Quantile Regression." NeurIPS, 2019. [Link](https://arxiv.org/abs/1905.03222)  
[3] Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. "Predictive inference with the jackknife+." Annals of Statistics, 2021. [Link](https://arxiv.org/abs/1905.02928)  

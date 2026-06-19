# Conformalized Quantile Regression (CQR): Theory & Implementation Summary

> **One-line description:** Conformalized Quantile Regression (CQR) wraps around standard quantile regression to provide mathematically guaranteed, distribution-free prediction intervals that adapt to the varying uncertainty (heteroscedasticity) of the data.

---

## 1. Overview & Intuition

Traditional conformal prediction approaches often output fixed-width prediction intervals. This happens when the underlying nonconformity score is based on the absolute residual of a point prediction (e.g., $|y - \hat{y}|$). In many real-world scenarios, the uncertainty of a prediction depends heavily on the input features (heteroscedasticity). For instance, predicting the price of a cheap car might have a narrow variance, whereas predicting the price of an expensive car might have a much wider variance. Fixed-width intervals fail to capture this input-dependent uncertainty, resulting in intervals that are too wide for easy cases and too narrow for hard ones.

Conformalized Quantile Regression (CQR) solves this by combining the adaptive nature of **quantile regression** with the rigorous finite-sample guarantees of **conformal prediction**. First, an underlying machine learning model is trained to predict the conditional quantiles of the target variable (e.g., the 5th and 95th percentiles). These conditional quantiles naturally adapt to the data's heteroscedasticity, forming an initial, variable-width prediction interval. 

However, quantile regression alone does not guarantee finite-sample marginal coverage. CQR addresses this by using a separate calibration set to compute a *normalized nonconformity score* based on how much the true targets deviate from the predicted quantile intervals. A quantile of these scores is then used to systematically stretch or shrink the initial prediction intervals. The resulting adjusted intervals are guaranteed to cover the true targets with the desired probability (e.g., 90%), while maintaining their adaptive, feature-dependent width.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

We are given a training set and a calibration set of $n$ independent and identically distributed (i.i.d.) examples. 
Let $X \in \mathbb{R}^d$ be the input feature space and $Y \in \mathbb{R}$ be the continuous output space.
Our goal is to construct a prediction interval $C(X_{test})$ for a new test point $X_{test}$ such that the true target $Y_{test}$ falls within the interval with probability at least $1 - \alpha$, where $\alpha \in (0,1)$ is a user-specified miscoverage rate.

We train a quantile regression model to estimate the conditional quantiles of $Y$ given $X$. Specifically, we obtain estimators for the lower and upper quantiles:
- $\hat{q}_{\alpha/2}(x)$ (e.g., 5th percentile for $\alpha=0.1$)
- $\hat{q}_{1-\alpha/2}(x)$ (e.g., 95th percentile for $\alpha=0.1$)

We denote the initial uncalibrated interval as $[\hat{l}(x), \hat{u}(x)]$, where $\hat{l}(x) = \hat{q}_{\alpha/2}(x)$ and $\hat{u}(x) = \hat{q}_{1-\alpha/2}(x)$.
We also define the interval width: $\hat{U}(x) = \hat{u}(x) - \hat{l}(x)$.

### 2.2 Core Definition / Score Function

To quantify how well the initial intervals fit the true data on the calibration set, CQR defines a nonconformity score. In the normalized variant commonly used (and seen in the notebook), the score is defined as:

**Equation:**
$$s_i = \frac{|y_i - \hat{y}_i|}{\hat{U}(x_i)}$$

**Where:**
- $s_i$ — The normalized nonconformity score for the $i$-th calibration sample.
- $y_i$ — The true target value.
- $\hat{y}_i$ — The median prediction (the conditional median $\hat{q}_{0.5}(x_i)$).
- $\hat{U}(x_i) = \hat{u}(x_i) - \hat{l}(x_i)$ — The predicted interval width (uncertainty) for the $i$-th sample.

**What this means:** This score measures the absolute prediction error scaled by the model's predicted uncertainty. If the model is highly uncertain (wide interval $\hat{U}$), a larger error is penalized less. *Note: The original CQR paper also proposes a signed maximum score $s_i = \max\{\hat{l}(x_i) - y_i, y_i - \hat{u}(x_i)\}$. The normalized absolute residual used in the notebook is an equivalent alternative often referred to as Conformalized Normalized Regression.*

### 2.3 Calibration Threshold and Prediction Set Construction

Using the calibration set of size $n$, we compute the nonconformity scores $s_1, \dots, s_n$. We then find the empirical quantile of these scores at a specific level to form the calibration threshold $\hat{q}$.

**Equation:**
$$\hat{q} = \text{Quantile}\left( \{s_1, \dots, s_n\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

**Where:**
- $\hat{q}$ — The calibration threshold.
- $n$ — The number of calibration samples.
- $1 - \alpha$ — The target coverage probability.

For a new test point $x_{test}$, the final calibrated prediction interval is constructed by expanding (or shrinking) the median prediction by the calibrated margin:

**Equation:**
$$C(x_{test}) = \left[ \hat{y}_{test} - \hat{q} \cdot \hat{U}(x_{test}), \;\; \hat{y}_{test} + \hat{q} \cdot \hat{U}(x_{test}) \right]$$

**What this means:** The adaptive interval width $\hat{U}(x_{test})$ is multiplied by the global scaling factor $\hat{q}$ and applied around the median prediction. This guarantees the $(1-\alpha)$ coverage while ensuring that the final interval width remains proportional to the local uncertainty $\hat{U}(x_{test})$.

---

## 3. Algorithm

**Input:** Data $(X_i, Y_i)_{i=1}^N$, test point $x_{test}$, miscoverage rate $\alpha$
**Output:** Prediction interval $C(x_{test})$

1. **Split Data:** Partition the available data into a proper training set and a calibration set of size $n$.
2. **Train Model:** Train a machine learning model on the proper training set to predict the conditional median $\hat{y}(x)$, the lower quantile $\hat{l}(x)$, and the upper quantile $\hat{u}(x)$.
3. **Compute Uncertainty:** For each sample in the calibration set, predict the interval bounds and compute the interval width $\hat{U}(x_i) = \hat{u}(x_i) - \hat{l}(x_i)$. (Avoid division by zero by setting $\hat{U}(x_i) = \epsilon$ if it is 0).
4. **Compute Scores:** For each sample in the calibration set, calculate the normalized nonconformity score: $s_i = |y_i - \hat{y}_i| / \hat{U}(x_i)$.
5. **Determine Threshold:** Calculate the quantile $\hat{q}$ of the calibration scores at the level $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$.
6. **Construct Interval:** For the test point $x_{test}$, compute $\hat{y}_{test}$ and $\hat{U}(x_{test})$. The final interval is $[\hat{y}_{test} - \hat{q} \hat{U}(x_{test}), \hat{y}_{test} + \hat{q} \hat{U}(x_{test})]$.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Quantile_Regression.ipynb`

### 4.1 Extracting Predictions and Splitting Sets
```python
lower = predictions_df[0.05].fillna(0)
upper = predictions_df[0.95].fillna(0)
pred = predictions_df[0.5].fillna(0)
labels = predictions_df['Actual'].fillna(0)

np.random.seed(random_seed)
idx = np.array([1] * n + [0] * (total_samples - n)) > 0
np.random.shuffle(idx)
cal_labels, val_labels = labels[idx], labels[~idx]
cal_upper, val_upper = upper[idx], upper[~idx]
cal_lower, val_lower = lower[idx], lower[~idx]
cal_pred, val_pred = pred[idx], pred[~idx]
```
**What this does:** The notebook loads pre-computed quantile predictions (5%, 50%, 95%) and actual labels from a DataFrame. It then splits the data into a calibration set of size $n$ and a validation (test) set.
**Why:** Conformal prediction requires a hold-out calibration set that the model hasn't seen during training to prevent over-fitting and to guarantee valid coverage mathematically.

### 4.2 Calculating Uncertainty Widths and Scores
```python
cal_U = cal_upper - cal_lower
val_U = val_upper - val_lower

cal_U[cal_U == 0] = np.finfo(float).eps
val_U[val_U == 0] = np.finfo(float).eps

cal_scores = np.abs(cal_pred - cal_labels) / cal_U
```
**What this does:** Computes the width of the predicted interval $\hat{U}$ for both calibration and validation sets. It replaces zeros with a tiny epsilon to avoid division-by-zero errors. Then, it calculates the normalized nonconformity score $s_i$ for the calibration set.
**Why:** The score normalizes the absolute error by the local uncertainty. This enables CQR to rescale the intervals multiplicatively later, preserving the adaptive (heteroscedastic) width property.

### 4.3 Computing the Calibration Threshold
```python
qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation='higher')
```
**What this does:** Computes the empirical quantile $\hat{q}$ of the calibration scores using the finite-sample adjusted quantile formula.
**Why:** The term $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ is the mathematically exact quantile level required by split conformal prediction theory to guarantee a marginal coverage probability of exactly $1-\alpha$ on new exchangeable test data.

### 4.4 Constructing and Evaluating the Calibrated Intervals
```python
prediction_sets = [val_pred - val_U * qhat, val_pred + val_U * qhat]

empirical_coverage_uncalibrated = ((val_labels >= prediction_sets_uncalibrated[0]) & (val_labels <= prediction_sets_uncalibrated[1])).mean() * 100
empirical_coverage = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean() * 100
```
**What this does:** Constructs the final prediction bounds by taking the median prediction $\pm$ the scaled uncertainty width. It then evaluates the empirical coverage percentage of both the raw (uncalibrated) quantile estimates and the CQR-calibrated intervals.
**Why:** This demonstrates that the raw quantile regression bounds often fail to achieve the nominal $1-\alpha$ coverage (e.g., they might only cover 85% instead of 90%). The calibrated CQR bounds will rigorously hit or exceed the 90% target.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

**Design rules for the toy dataset:**
- Target problem: Regression
- Number of calibration samples $n = 5$
- Target coverage: $1-\alpha = 0.80$ ($\alpha = 0.20$)

Let the true values and model predictions for the 5 calibration samples be:

| Sample $i$ | True $y_i$ | $\hat{l}_i$ (5%) | $\hat{y}_i$ (50%) | $\hat{u}_i$ (95%) | Uncalibrated Interval |
|---|---|---|---|---|---|
| 1 | 12.0 | 9.0 | 11.0 | 13.0 | [9.0, 13.0] |
| 2 | 25.0 | 18.0 | 22.0 | 26.0 | [18.0, 26.0] |
| 3 | 9.0 | 6.0 | 8.0 | 10.0 | [6.0, 10.0] |
| 4 | 33.0 | 28.0 | 30.0 | 32.0 | [28.0, 32.0] |
| 5 | 17.0 | 16.0 | 18.0 | 20.0 | [16.0, 20.0] |

We have 3 test samples:
- **Easy Sample:** Model perfectly predicts median, narrow width.
- **Borderline Sample:** Target is right on the edge of the uncalibrated interval.
- **Ambiguous Sample:** Model is highly uncertain, very wide width, large error.

| Test Sample | True $y_{test}$ | $\hat{l}_{test}$ (5%) | $\hat{y}_{test}$ (50%) | $\hat{u}_{test}$ (95%) |
|---|---|---|---|---|
| A (Easy) | 15.0 | 14.0 | 15.0 | 16.0 |
| B (Borderline)| 44.0 | 38.0 | 40.0 | 43.0 |
| C (Ambiguous)| 60.0 | 45.0 | 50.0 | 55.0 |

---

### 5.1 Method: Conformalized Quantile Regression (Normalized)

#### Step A — Compute nonconformity scores on the calibration set

Calculate the interval width $\hat{U}_i = \hat{u}_i - \hat{l}_i$, and the nonconformity score $s_i = \frac{|y_i - \hat{y}_i|}{\hat{U}_i}$.

| Sample $i$ | True $y_i$ | $\hat{y}_i$ | $\hat{U}_i = \hat{u}_i - \hat{l}_i$ | Absolute Error $|y_i - \hat{y}_i|$ | Score $s_i$ |
|---|---|---|---|---|---|
| 1 | 12.0 | 11.0 | 13.0 - 9.0 = 4.0 | 1.0 | 1.0 / 4.0 = **0.250** |
| 2 | 25.0 | 22.0 | 26.0 - 18.0 = 8.0 | 3.0 | 3.0 / 8.0 = **0.375** |
| 3 | 9.0  | 8.0  | 10.0 - 6.0 = 4.0 | 1.0 | 1.0 / 4.0 = **0.250** |
| 4 | 33.0 | 30.0 | 32.0 - 28.0 = 4.0 | 3.0 | 3.0 / 4.0 = **0.750** |
| 5 | 17.0 | 18.0 | 20.0 - 16.0 = 4.0 | 1.0 | 1.0 / 4.0 = **0.250** |

Sorted Scores: `[0.250, 0.250, 0.250, 0.375, 0.750]`

#### Step B — Compute the calibration threshold

We want $1-\alpha = 0.80$. The number of calibration samples is $n = 5$.
The quantile level index calculation:
$$q_\text{level} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n} = \frac{\lceil (5+1)(0.80) \rceil}{5} = \frac{\lceil 6 \times 0.80 \rceil}{5} = \frac{\lceil 4.8 \rceil}{5} = \frac{5}{5} = 1.0$$

Since the level is 1.0, the threshold $\hat{q}$ is simply the maximum score in the calibration set.
$$\hat{q} = 0.750$$

*(Note: because $n$ is so small, an 80% coverage target effectively requires us to use the 100th percentile of the calibration scores).*

#### Step C — Build prediction sets for each test sample

For each test sample, compute the interval width $\hat{U}_{test} = \hat{u}_{test} - \hat{l}_{test}$.
The calibrated interval is $[\hat{y}_{test} - \hat{q} \hat{U}_{test}, \hat{y}_{test} + \hat{q} \hat{U}_{test}]$.

**Test Sample A (Easy):**
- $\hat{U}_A = 16.0 - 14.0 = 2.0$
- Margin = $\hat{q} \times \hat{U}_A = 0.750 \times 2.0 = 1.5$
- Interval: $[15.0 - 1.5, 15.0 + 1.5] = \mathbf{[13.5, 16.5]}$
- Uncalibrated interval was $[14.0, 16.0]$. True $y_A = 15.0$. Covered? ✓

**Test Sample B (Borderline):**
- $\hat{U}_B = 43.0 - 38.0 = 5.0$
- Margin = $\hat{q} \times \hat{U}_B = 0.750 \times 5.0 = 3.75$
- Interval: $[40.0 - 3.75, 40.0 + 3.75] = \mathbf{[36.25, 43.75]}$
- Uncalibrated interval was $[38.0, 43.0]$. True $y_B = 44.0$. Covered? ✗ (Barely missed).

**Test Sample C (Ambiguous):**
- $\hat{U}_C = 55.0 - 45.0 = 10.0$
- Margin = $\hat{q} \times \hat{U}_C = 0.750 \times 10.0 = 7.5$
- Interval: $[50.0 - 7.5, 50.0 + 7.5] = \mathbf{[42.5, 57.5]}$
- Uncalibrated interval was $[45.0, 55.0]$. True $y_C = 60.0$. Covered? ✗.

#### Step D — Summary table for this method

| Test Sample | True $Y$ | Uncalibrated Set | Calibrated Set | Set Width | Coverage |
|---|---|---|---|---|---|
| A (Easy) | 15.0 | [14.0, 16.0] | [13.5, 16.5] | 3.0 | ✓ |
| B (Borderline) | 44.0 | [38.0, 43.0] | [36.25, 43.75] | 7.5 | ✗ |
| C (Ambiguous) | 60.0 | [45.0, 55.0] | [42.5, 57.5] | 15.0 | ✗ |

*Notice how the calibrated set width directly scales with the model's initial uncertainty (width of 3.0 for sample A, and 15.0 for sample C), demonstrating the adaptive, heteroscedastic nature of CQR.*

---

## 6. References

[1] Romano, Yaniv, Evan Patterson, and Emmanuel Candes. "Conformalized quantile regression." Advances in neural information processing systems 32 (2019). [Link](https://arxiv.org/abs/1905.03222)
[2] Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. "Algorithmic Learning in a Random World." Springer Science & Business Media (2005). [Link](https://link.springer.com/book/10.1007/b138466)

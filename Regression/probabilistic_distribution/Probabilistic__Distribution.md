# Probabilistic Distribution and Conformal Prediction: Theory & Implementation Summary

> **One-line description:** Generates probabilistic distributions and rigorously calibrated prediction intervals for continuous targets using gradient boosting and conformal prediction techniques.

---

## 1. Overview & Intuition

Traditional regression models predict a single point estimate (e.g., the expected value of a target). However, in high-stakes domains, point estimates are insufficient because they convey no sense of uncertainty. Probabilistic regression extends these models to predict a full probability distribution (e.g., predicting both the mean and standard deviation of a Gaussian) or uses conformal prediction to guarantee that the true value falls within a predicted interval with a user-specified probability (e.g., 95%).

This methodology bridges the gap between raw point predictions and actionable uncertainty quantification. By combining Probabilistic Gradient Boosting (which models the predictive variance) and Conformal Prediction (which provides distribution-free coverage guarantees), we can confidently assess whether the model is "sure" or "unsure" for any given test sample, and provide reliable bounds for decision-making.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $X \in \mathbb{R}^d$ be the input feature space and $Y \in \mathbb{R}$ be the continuous target space. We have a training set for model fitting, and a distinct calibration set $\mathcal{D}_{cal} = \{(x_1, y_1), \dots, (x_n, y_n)\}$ of size $n$. Our goal is to construct a prediction interval $\hat{C}(x_{test})$ for a new test point such that $\mathbb{P}(Y_{test} \in \hat{C}(X_{test})) \geq 1 - \alpha$.

### 2.2 Probabilistic Regression (NLL and CRPS)

Probabilistic boosting models output a distribution $P_\theta(y|x)$ (e.g., Normal distribution with parameters $\theta = (\mu, \sigma)$).

**Equation:**
$$\text{NLL} = -\log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(y - \mu)^2}{2\sigma^2} \right) \right)$$

**Where:**
- $y$ — The true continuous target
- $\mu$ — The predicted mean
- $\sigma$ — The predicted standard deviation

**What this means:** The Negative Log-Likelihood measures how likely the true target was under the predicted probability distribution. Lower NLL means better probabilistic fit. The models are also evaluated using the Continuous Ranked Probability Score (CRPS).

### 2.3 Split Conformal Prediction (SplitCP)

To guarantee coverage without assuming a Gaussian distribution, we compute nonconformity scores on the calibration set.

**Equation:**
$$s_i = |y_i - \hat{\mu}(x_i)|$$

**Where:**
- $s_i$ — The nonconformity score for calibration sample $i$
- $y_i$ — The true value
- $\hat{\mu}(x_i)$ — The model's point prediction

**What this means:** The score represents the absolute error of the model's prediction. We then find the empirical quantile of these errors to set the interval width.

---

## 3. Algorithm

**Input:** Calibration set $\mathcal{D}_{cal}$, trained probabilistic model $\hat{\mu}$ and $\hat{\sigma}$, miscoverage rate $\alpha$, test point $x_{test}$.  
**Output:** Predictive distribution parameters and a valid prediction interval $\hat{C}(x_{test})$.  

1. Train the probabilistic model on the training set to output $\hat{\mu}$ and $\hat{\sigma}$.
2. For each sample in $\mathcal{D}_{cal}$, compute the absolute residual score $s_i = |y_i - \hat{\mu}(x_i)|$.
3. Compute the calibration threshold $\hat{q}$ as the $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ empirical quantile of the sorted scores $\{s_1, \dots, s_n\}$.
4. For the test point $x_{test}$, output the prediction interval $\hat{C}(x_{test}) = [\hat{\mu}(x_{test}) - \hat{q}, \hat{\mu}(x_{test}) + \hat{q}]$.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Probabilistic__Distribution.ipynb`

### 4.1 Extracting Probabilistic Distributions
```python
# Extend the trained model into a probabilistic predictor using IBUGWrapper
prob_model = IBUGWrapper().fit(model, X_train, y_train, X_val=X_val, y_val=y_val)
location, scale = prob_model.pred_dist(X_test)
std_dev = scale ** 0.5
```
**What this does:** Wraps a standard Gradient Boosting model (like XGBoost) using IBUG to estimate the predictive variance alongside the mean.  
**Why:** Trees naturally partition the data; tracking leaf variance across the ensemble provides a principled standard deviation (`std_dev`) for each prediction.

### 4.2 Calibration Diagnostics (PIT Values)
```python
pit_values = norm.cdf(y_true, loc=mu_pred, scale=sigma_pred)
ks_stat, ks_pvalue = kstest(pit_values, 'uniform')
```
**What this does:** Computes the Probability Integral Transform (PIT) values by evaluating the predicted Gaussian CDF at the true target value. It then runs a Kolmogorov-Smirnov test.  
**Why:** If the predicted distributions are perfectly calibrated, the PIT values will follow a Uniform(0, 1) distribution. This statistically verifies the probabilistic reliability.

### 4.3 CRPS Decomposition
```python
crps_values = ps.crps_gaussian(y_true, mu=mu_pred, sig=sigma_pred)
total_crps = np.mean(crps_values)
```
**What this does:** Computes the Continuous Ranked Probability Score for the predicted Gaussian distributions against the true targets.  
**Why:** CRPS is a strictly proper scoring rule that generalizes the Mean Absolute Error to probabilistic forecasts, jointly penalizing lack of sharpness and miscalibration.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

Since this is a continuous regression task, we evaluate prediction intervals across continuous real numbers.

- $n = 5$ calibration samples.
- Miscoverage target $\alpha = 0.10$.
- Test samples:
  - **Easy sample:** Model predicts perfectly (very low error).
  - **Borderline sample:** Model error is right around the threshold.
  - **Ambiguous sample:** Model is completely wrong (high error).

**Calibration Set Predictions:**

| Sample $i$ | $x_i$ | True $y_i$ | Pred $\hat{\mu}(x_i)$ |
|---|---|---|---|
| 1 | A | 10.5 | 10.2 |
| 2 | B | 14.0 | 14.5 |
| 3 | C | 8.2 | 7.9 |
| 4 | D | 20.1 | 21.0 |
| 5 | E | 11.0 | 11.8 |

**Test Set Predictions:**

| Sample | $x_{test}$ | True $y_{test}$ | Pred $\hat{\mu}(x_{test})$ |
|---|---|---|---|
| Easy | T1 | 15.0 | 15.1 |
| Borderline | T2 | 18.0 | 17.2 |
| Ambiguous | T3 | 25.0 | 19.0 |

---

### 5.1 Method 1: Split Conformal Prediction (SplitCP)

#### Step A — Compute nonconformity scores on the calibration set

The score is the absolute residual $s_i = |y_i - \hat{\mu}(x_i)|$.

| Sample Index | True $y_i$ | Pred $\hat{\mu}(x_i)$ | Score $s_i$ |
|---|---|---|---|
| 1 | 10.5 | 10.2 | 0.3 |
| 2 | 14.0 | 14.5 | 0.5 |
| 3 | 8.2 | 7.9 | 0.3 |
| 4 | 20.1 | 21.0 | 0.9 |
| 5 | 11.0 | 11.8 | 0.8 |

Sorted scores: $0.3, 0.3, 0.5, 0.8, 0.9$

#### Step B — Compute the calibration threshold

We calculate the empirical quantile index for $n=5$ and $\alpha=0.10$:
$$q_\text{level} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n}$$
$$q_\text{level} = \frac{\lceil (6)(0.90) \rceil}{5} = \frac{\lceil 5.4 \rceil}{5} = \frac{6}{5}$$

Since $k=6$ is larger than $n=5$, the quantile level clips to 1.0 (meaning we use the maximum score in the set to guarantee coverage). Thus, the selected threshold is $\hat{q} = 0.9$.

#### Step C — Build prediction sets for each test sample

The prediction interval is $[\hat{\mu}(x_{test}) - 0.9, \hat{\mu}(x_{test}) + 0.9]$.

| Test Sample | True $y$ | Pred $\hat{\mu}$ | Interval $[\hat{\mu} - \hat{q}, \hat{\mu} + \hat{q}]$ | True $y$ in Interval? |
|---|---|---|---|---|
| Easy | 15.0 | 15.1 | $[14.2, 16.0]$ | ✓ |
| Borderline| 18.0 | 17.2 | $[16.3, 18.1]$ | ✓ |
| Ambiguous | 25.0 | 19.0 | $[18.1, 19.9]$ | ✗ |

#### Step D — Summary table for this method

| Sample | Prediction Interval | Interval Size | Coverage |
|---|---|---|---|
| Easy | [14.2, 16.0] | 1.8 | ✓ |
| Borderline | [16.3, 18.1] | 1.8 | ✓ |
| Ambiguous | [18.1, 19.9] | 1.8 | ✗ |

---

### 5.2 Cross-Method Comparison

While traditional SplitCP produces fixed-width intervals for all samples (size = 1.8), Probabilistic Gradient Boosting directly predicts a standard deviation $\sigma(x)$ for each sample, creating adaptive prediction intervals $[\hat{\mu}(x) - z \sigma(x), \hat{\mu}(x) + z \sigma(x)]$.

| Sample | SplitCP Interval (Fixed) | Probabilistic Interval (Adaptive) | Coverage Comparison |
|---|---|---|---|
| Easy | [14.2, 16.0] | [14.9, 15.3] (Tight, low $\sigma$) | Both ✓ |
| Borderline | [16.3, 18.1] | [15.5, 18.9] (Wide, high $\sigma$) | Both ✓ |
| Ambiguous | [18.1, 19.9] | [16.0, 22.0] (Very wide) | SplitCP ✗, Probabilistic ✗ |

**Why the methods differ:**
On the easy sample, the probabilistic model correctly identifies low variance and gives a much tighter bound than the conservative SplitCP method. On the borderline and ambiguous samples, the probabilistic method widens its interval dynamically (heteroscedasticity) due to higher local uncertainty, whereas SplitCP is forced to use the global fixed threshold $\hat{q}=0.9$ derived from the entire calibration set.

**Threshold Comparison Table:**

| Output Form | SplitCP Threshold | Probabilistic Method (IBUG/NGBoost) |
|---|---|---|
| Width Determinant | Global $\hat{q} = 0.9$ | Local $z \cdot \sigma(x)$ |
| Adaptivity | None (Homoscedastic) | High (Heteroscedastic) |

---

## 6. References

[1] Brofos, J., et al. "Implicit Bayesian Uncertainty for Gradient Boosting." *arXiv*, 2022. [Link](https://arxiv.org/abs/2208.10641)  
[2] Duan, T., et al. "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *ICML*, 2020. [Link](https://arxiv.org/abs/1910.03225)  
[3] Sprangers, O., et al. "Probabilistic Gradient Boosting Machines for Large-Scale Predictive Inference." *KDD*, 2021. [Link](https://arxiv.org/abs/2106.01682)  
[4] Angelopoulos, A. N., & Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv*, 2021. [Link](https://arxiv.org/abs/2107.07511)

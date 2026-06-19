# Adaptive Coverage Policies (ACP): Theory & Implementation Summary

> **One-line description:** Adaptive Coverage Policies (ACP) treats the miscoverage rate $\alpha$ as a learnable parameter, optimizing a differentiable proxy of the interval size penalized by $\alpha$ to find a dataset-specific optimal confidence level.

---

## 1. Overview & Intuition

Standard conformal prediction provides distribution-free marginal coverage guarantees for a fixed, user-specified miscoverage rate $\alpha$ (e.g., $\alpha = 0.1$). However, for certain datasets and models, blindly fixing $\alpha$ can yield excessively wide and uninformative prediction intervals. Setting $\alpha$ manually often involves guesswork or trial and error.

Adaptive Coverage Policies (ACP) aims to eliminate this guesswork by learning the optimal $\alpha$ dynamically. Instead of taking $\alpha$ as an input, ACP treats it as a learnable parameter output by a small neural network (`AlphaNet`). The network looks at the sum of the nonconformity errors from the calibration set and balances the desire for small, tight prediction intervals against a regularization penalty on $\alpha$. This enables the method to automatically choose a confidence level that avoids catastrophically large intervals while preserving as much coverage as possible.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $X$ be the input space and $Y = \mathbb{R}$ the output space for a regression task. We are given a pre-trained regression model $\hat{f}$ and a calibration dataset of $n$ samples, $D_{\text{cal}} = \{(X_i, y_i)\}_{i=1}^n$. 

### 2.2 Nonconformity Score

For regression tasks, the standard nonconformity score is the absolute residual error of the model's predictions.

**Equation:**
$$s_i = | \hat{f}(X_i) - y_i |$$

**Where:**
- $s_i$ — the nonconformity score for the $i$-th calibration sample
- $\hat{f}(X_i)$ — the predicted continuous target
- $y_i$ — the true continuous target

**What this means:** It captures the magnitude of the model's prediction error. Larger scores indicate higher uncertainty or worse model fit for that sample.

### 2.3 Differentiable Interval Size Proxy

To optimize $\alpha$ via gradient descent, ACP cannot use the standard non-differentiable quantile selection (i.e., sorting and picking the $k$-th score). Instead, it formulates a continuous, differentiable proxy for the interval width:

**Equation:**
$$\text{size}(\alpha) = \frac{2 \sum_{i=1}^n s_i}{\max(\alpha(n+1) - 1, \epsilon)}$$

**Where:**
- $\text{size}(\alpha)$ — the estimated width of the prediction interval
- $\sum_{i=1}^n s_i$ — the sum of all absolute nonconformity scores
- $\alpha$ — the adaptive miscoverage rate ($\alpha \in (0,1)$)
- $\epsilon$ — a small numerical constant to prevent division by zero

**What this means:** This function maps the chosen $\alpha$ to an expected interval width. As the miscoverage rate $\alpha$ approaches zero (requiring higher coverage), the denominator shrinks, causing the interval size proxy to grow rapidly.

### 2.4 Optimization Objective

The optimal coverage parameter $\tilde{\alpha}$ is predicted by a multi-layer perceptron (`AlphaNet`) parameterized by $\theta$. The network is trained using a leave-one-out strategy on the calibration set to minimize:

**Equation:**
$$\mathcal{L}(\theta) = \text{size}(\tilde{\alpha}) + \lambda \tilde{\alpha}$$

**Where:**
- $\mathcal{L}(\theta)$ — the objective loss function
- $\text{size}(\tilde{\alpha})$ — the continuous proxy for the interval width
- $\lambda$ — a regularization hyperparameter balancing interval width and confidence
- $\tilde{\alpha} = \text{AlphaNet}_{\theta}(F)$ — the predicted alpha, bounded in $(0, 1)$

**What this means:** The loss function encourages tighter intervals while penalizing large values of $\tilde{\alpha}$. A larger $\tilde{\alpha}$ would artificially shrink the intervals at the cost of losing significant coverage.

---

## 3. Algorithm

**Input:** Base regression model $\hat{f}$, calibration set $D_{\text{cal}}$, test sample $x_{\text{test}}$, regularization sequence $\{\lambda_k\}$.  
**Output:** Adaptive coverage parameter $\hat{\alpha}$, prediction interval $[L, U]$.

1. Split the training data into a fitting set and a calibration set $D_{\text{cal}}$.
2. Train the base regression model $\hat{f}$ on the fitting set.
3. Compute nonconformity scores $s_i = |\hat{f}(X_i) - y_i|$ for all $i \in D_{\text{cal}}$.
4. Construct leave-one-out feature representations $F_j = \sum_{i \neq j} s_i$ for each calibration sample.
5. Train `AlphaNet` on $\{F_j\}$ over multiple $\lambda$ values to minimize the regularized interval size loss. Select the model yielding the lowest final loss.
6. Compute the full calibration feature $F = \sum_{i=1}^n s_i$.
7. Evaluate `AlphaNet` to predict the final optimal coverage level: $\hat{\alpha} = \text{AlphaNet}(F)$.
8. Calculate the global interval width: $W = \text{size}(\hat{\alpha})$.
9. Output the prediction set for any test sample: $[\hat{f}(x_{\text{test}}) - \frac{W}{2}, \hat{f}(x_{\text{test}}) + \frac{W}{2}]$.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Adaptive_Coverage_Policies(ACP).ipynb`

### 4.1 AlphaNet Architecture
```python
class AlphaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
```
**What this does:** A PyTorch multi-layer perceptron that takes the scalar sum of calibration errors (`input_dim=1`) and outputs a continuous value. A final Sigmoid bounds the output to $(0,1)$.
**Why:** It acts as the learned policy function mapping the overall dataset difficulty directly to the optimal miscoverage rate.

### 4.2 Differentiable Interval Size Computation
```python
def interval_size(scores, alpha, eps=1e-6):
    n = scores.numel()
    denominator = torch.clamp(alpha * (n + 1) - 1.0, min=eps)
    return 2.0 * scores.sum() / denominator
```
**What this does:** Implements the continuous proxy formula for the conformal interval width.
**Why:** Traditional non-parametric conformal prediction relies on sorting arrays and extracting discrete quantiles. Since sorting is non-differentiable, this continuous function allows gradients to flow backwards through $\alpha$ to update the `AlphaNet` weights.

### 4.3 Evaluation on Test Set
```python
    calib_scores = np.abs(model.predict(X_calib_np) - y_calib_np)
    test_feature = torch.tensor([[calib_scores.sum()]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        alpha_hat = float(torch.clamp(alpha_net(test_feature), min=1e-3, max=1.0 - 1e-3).item())
        
    score_tensor = torch.tensor(calib_scores, dtype=torch.float32, device=DEVICE)
    interval_width = float(interval_size(score_tensor, torch.tensor(alpha_hat, device=DEVICE)).item())
```
**What this does:** Evaluates `AlphaNet` on the full calibration score sum to yield $\hat{\alpha}$, then calculates a fixed, global interval width `interval_width`.
**Why:** Under this implementation, the adaptive policy learns an optimal global interval size for the entire dataset distribution rather than an individual width per test sample. The width is symmetric and uniformly applied around the point predictions $\hat{y}_{\text{test}}$.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

*(Note: The `SKILL.md` requires $K$ classes for a classification problem, but since this notebook strictly implements a **Regression** methodology, this example adapts the format to continuous targets while maintaining the 3 required test samples: easy, borderline, and ambiguous).*

**Toy Regression Calibration Dataset ($n=5$):**
- Sample 1: $\hat{y}_1 = 10.0, y_1 = 10.2$ (Error: $0.2$)
- Sample 2: $\hat{y}_2 = 15.0, y_2 = 14.5$ (Error: $0.5$)
- Sample 3: $\hat{y}_3 = 20.0, y_3 = 21.0$ (Error: $1.0$)
- Sample 4: $\hat{y}_4 = 25.0, y_4 = 24.1$ (Error: $0.9$)
- Sample 5: $\hat{y}_5 = 30.0, y_5 = 31.5$ (Error: $1.5$)

**Test samples:**
- **Easy sample:** $x_E$ where $\hat{y}_E = 12.0$ (High confidence region)
- **Borderline sample:** $x_B$ where $\hat{y}_B = 18.0$ (Medium confidence region)
- **Ambiguous sample:** $x_A$ where $\hat{y}_A = 28.0$ (Low confidence/high variance region)

**Hyperparameters:**
- Regularization $\lambda = 10$.
- Alpha $\alpha$ is adaptive.

---

### 5.1 Method 1: Adaptive Coverage Policies (ACP)

#### Step A — Compute nonconformity scores on the calibration set

| Sample index | True $y_i$ | Predicted $\hat{y}_i$ | Score $s_i$ |
|---|---|---|---|
| 1 | 10.2 | 10.0 | 0.2 |
| 2 | 14.5 | 15.0 | 0.5 |
| 3 | 21.0 | 20.0 | 1.0 |
| 4 | 24.1 | 25.0 | 0.9 |
| 5 | 31.5 | 30.0 | 1.5 |

Full list of calibration scores: `[0.2, 0.5, 1.0, 0.9, 1.5]`
Sum of calibration scores: $\sum s_i = 4.1$

#### Step B — Compute the calibration threshold (Optimal Alpha)

The `AlphaNet` takes the feature $F = 4.1$ and outputs a predicted $\tilde{\alpha}$. For this numerical example, assume the converged network outputs $\tilde{\alpha} = 0.30$.
Using the differentiable interval size proxy:
$$ \text{denominator} = \tilde{\alpha}(n+1) - 1.0 = 0.30(6) - 1.0 = 1.8 - 1.0 = 0.8 $$
$$ W = \text{size}(0.30) = \frac{2 \sum s_i}{\text{denominator}} = \frac{2(4.1)}{0.8} = \frac{8.2}{0.8} = 10.25 $$
The global prediction interval width is set to $10.25$.

#### Step C — Build prediction sets for each test sample

Because ACP in this implementation yields a dataset-level optimal width, the same width is applied symmetrically across all test predictions: $[\hat{y} - 5.125, \hat{y} + 5.125]$.

1. **Easy Sample ($x_E$)**:
   - Prediction: $12.0$
   - Set: $[12.0 - 5.125, 12.0 + 5.125] = [6.875, 17.125]$
2. **Borderline Sample ($x_B$)**:
   - Prediction: $18.0$
   - Set: $[18.0 - 5.125, 18.0 + 5.125] = [12.875, 23.125]$
3. **Ambiguous Sample ($x_A$)**:
   - Prediction: $28.0$
   - Set: $[28.0 - 5.125, 28.0 + 5.125] = [22.875, 33.125]$

#### Step D — Summary table for this method

| Test Sample | Point Prediction $\hat{y}$ | Predicted $\tilde{\alpha}$ | Global Width | Prediction Set |
|---|---|---|---|---|
| Easy ($x_E$) | 12.0 | 0.30 | 10.25 | $[6.875, 17.125]$ |
| Borderline ($x_B$) | 18.0 | 0.30 | 10.25 | $[12.875, 23.125]$ |
| Ambiguous ($x_A$) | 28.0 | 0.30 | 10.25 | $[22.875, 33.125]$ |

---

### 5.Z Cross-Method Comparison

Since the notebook implements a single uncertainty quantification pipeline applied over various gradient boosting algorithms, a cross-method comparison within the CP framework highlights Standard Split Conformal Prediction vs Adaptive Coverage Policies (ACP):

| Test Sample | Standard CP ($\alpha=0.10$) Set | ACP ($\lambda=10$) Set | Width Diff |
|---|---|---|---|
| Easy | $[10.5, 13.5]$ | $[6.875, 17.125]$ | +7.25 |
| Borderline | $[16.5, 19.5]$ | $[12.875, 23.125]$ | +7.25 |
| Ambiguous | $[26.5, 29.5]$ | $[22.875, 33.125]$ | +7.25 |

**Why the methods differ:**
- **Standard CP** rigidly adheres to the predetermined $\alpha=0.10$. With $n=5$, the required quantile index is $\lceil 6(0.9) \rceil = 6$. Since we only have 5 samples, it clips to the maximum score, yielding a threshold of $1.5$, giving a width of $2(1.5) = 3.0$.
- **ACP** acts adaptively. Penalized by $\lambda=10$, `AlphaNet` settled on an optimal $\tilde{\alpha}=0.30$. Using the analytical proxy size formula $\frac{8.2}{0.8}$, it generated a wider interval width of $10.25$. Depending on the hyperparameter $\lambda$ and dataset size, ACP will intelligently shift this width to find the optimal global compromise.

---

## 6. References

[1] Vovk, V., Gammerman, A., & Shafer, G. "Algorithmic Learning in a Random World." Springer, 2005. [Link](https://link.springer.com/book/10.1007/b106715)  
[2] Angelopoulos, A. N., & Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." arXiv, 2021. [Link](https://arxiv.org/abs/2107.07511)

# Hyperspherical Confidence Mapping (HCM): Theory & Implementation Summary

> **One-line description:** A regression uncertainty method that maps 1D targets to higher-dimensional hyperspheres to independently predict target magnitude and direction, using the predicted direction's norm deviation to estimate predictive uncertainty.

---

## 1. Overview & Intuition

Hyperspherical Confidence Mapping (HCM) is a deterministic, sampling-free framework for uncertainty estimation in neural networks. Traditional regression uncertainty methods often rely on computationally expensive Bayesian ensembles, Monte Carlo dropout, or complex quantile regression losses. In contrast, HCM provides a direct estimation of uncertainty within a single forward pass without requiring explicit distributional assumptions.

The core insight behind HCM is to reframe the scalar regression task as a geometric learning problem. A scalar target is artificially expanded into a higher-dimensional embedding (for example, duplicating the scalar to create a 2D vector). The model is then trained to predict both the magnitude and the normalized direction of this expanded vector separately. Because the direction vector is geometrically constrained to lie on a unit hypersphere, any deviation from this constraint (i.e., when the predicted direction's norm is not exactly 1) serves as a direct indicator of uncertainty. When the model is uncertain due to noisy or sparse data, it struggles to confidently place the direction vector on the hypersphere, causing the norm to shrink. This violation of the geometric constraint is then calibrated to produce reliable predictive uncertainty bounds.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Consider a regression dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with input features $\mathbf{x}_i \in \mathbb{R}^d$ and continuous scalar targets $y_i \in \mathbb{R}$. We want to predict $\hat{y}$ along with a reliable uncertainty measure $\hat{\sigma}$.

The first step is target expansion. The scalar $y$ is mapped to a $D$-dimensional space (e.g., $D=2$):
$$\mathbf{y}_{ext} = [y, y]^T \in \mathbb{R}^D$$

### 2.2 Geometric Decomposition

The expanded target is decomposed into its true magnitude and its true unit direction:

**Equation:**
$$R_{target} = ||\mathbf{y}_{ext}||_2 = \sqrt{D \cdot y^2}$$
$$\mathbf{d}_{target} = \frac{\mathbf{y}_{ext}}{R_{target} + \epsilon}$$

**Where:**
- $R_{target}$ — The Euclidean norm (magnitude) of the expanded vector.
- $\mathbf{d}_{target}$ — The normalized direction vector lying on the unit hypersphere.
- $\epsilon$ — A small constant to prevent division by zero.

**What this means:** Instead of predicting $y$ directly, the model must predict its geometric components: the overall scale ($R$) and its orientation ($\mathbf{d}$). 

### 2.3 Symmetric Loss Function

The model outputs predictions $\hat{R} \in \mathbb{R}$ and $\hat{\mathbf{d}} \in \mathbb{R}^D$. The loss function symmetrically penalizes errors in both components:

**Equation:**
$$\mathcal{L}_{HCM} = \text{MSE}(\hat{R} \cdot \mathbf{d}_{target}, \mathbf{y}_{ext}) + \text{MSE}(R_{target} \cdot \hat{\mathbf{d}}, \mathbf{y}_{ext})$$

**Where:**
- $\text{MSE}$ — Mean Squared Error loss.
- $\hat{R} \cdot \mathbf{d}_{target}$ — The predicted magnitude paired with the ground-truth direction.
- $R_{target} \cdot \hat{\mathbf{d}}$ — The predicted direction paired with the ground-truth magnitude.

**What this means:** This loss ensures that both the magnitude branch and the direction branch learn to independently reconstruct the target.

### 2.4 Uncertainty Extraction

During inference, the model produces a raw uncertainty score based on how much the predicted direction deviates from being a valid unit vector:

**Equation:**
$$\hat{\sigma}_{raw} = \sqrt{ \left| ||\hat{\mathbf{d}}||_2^2 - 1 \right| } \cdot |\hat{R}|$$

**Where:**
- $||\hat{\mathbf{d}}||_2^2$ — The squared norm of the predicted direction vector.
- $1$ — The expected squared norm of a true unit direction vector.
- $\hat{R}$ — The predicted magnitude, which scales the uncertainty to match the target's scale.

**What this means:** If the model is confident, $||\hat{\mathbf{d}}||_2 \approx 1$, and uncertainty is near zero. If the model is uncertain, the norm shrinks, increasing the absolute deviation from 1 and producing a larger uncertainty bound proportional to the target's scale.

---

## 3. Algorithm

**Input:** Training set $\mathcal{D}_{train}$, Calibration set $\mathcal{D}_{cal}$, target confidence levels (e.g., 68%, 95%, 99.7%), scaling grid $G$.  
**Output:** Trained HCM model, scalar temperature $s$.

1. **Target Expansion:** For each sample, expand the 1D target $y$ to a 2D vector $\mathbf{y}_{ext} = [y, y]^T$.
2. **Model Training:** Train a dual-head neural network to output magnitude $\hat{R}$ and direction $\hat{\mathbf{d}}$. Optimize the network using the symmetric $\mathcal{L}_{HCM}$ loss.
3. **Inference & Raw Uncertainty:** On the calibration set $\mathcal{D}_{cal}$, extract the mean prediction $\hat{y} = (\hat{R} \cdot \hat{\mathbf{d}})_0$, the absolute error $|\hat{y} - y|$, and the raw uncertainty $\hat{\sigma}_{raw}$.
4. **Temperature Scaling (Calibration):** Perform a grid search over $s \in G$ to find the temperature scale that minimizes the squared difference between the empirical coverage of $k \cdot (s \cdot \hat{\sigma}_{raw})$ and the target theoretical Gaussian coverage (68% for $1\sigma$, 95% for $2\sigma$, 99.7% for $3\sigma$).
5. **Prediction:** For any new test sample, compute the mean prediction $\hat{y}$ and the calibrated standard deviation $\hat{\sigma} = s \cdot \hat{\sigma}_{raw}$.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Hyperspherical_Confidence_Mapping(HCM).ipynb`

### 4.1 Target Expansion
```python
def expand_scalar_targets(y_array):
    y_array = np.asarray(y_array, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([y_array, y_array], axis=1)
```
**What this does:** Duplicates the 1D regression target to form a 2D vector for every sample in the batch.  
**Why:** This embedding into a higher dimensional space is required to compute a meaningful direction vector and its corresponding norm on a unit hypersphere.

### 4.2 Network Architecture
```python
class HCMUCIRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)   # direction d
        self.fc5 = nn.Linear(20, 1)   # magnitude R
...
```
**What this does:** Defines a multi-layer perceptron with a shared backbone that splits into two heads: a 2-dimensional output for the direction $\mathbf{d}$ and a 1-dimensional output for the magnitude $R$.  
**Why:** This dual-head structure naturally accommodates the geometric decomposition central to HCM.

### 4.3 Symmetric Loss Computation
```python
R_target = torch.sqrt(torch.sum(y_expanded ** 2, dim=1, keepdim=True))
d_target = y_expanded / (R_target + 1e-8)

d_loss = criterion(pred_R * d_target, y_expanded)
R_loss = criterion(R_target * pred_d, y_expanded)
loss = d_loss + R_loss
```
**What this does:** Calculates the true magnitude and direction, crosses them with the model's predicted magnitude and direction, computes their MSE against the expanded target, and sums them.  
**Why:** It enforces that both the magnitude prediction and the direction prediction independently encode enough information to reconstruct the target.

### 4.4 Uncertainty Extraction
```python
d_norm_sq = torch.sum(pred_d ** 2, dim=1)
d_norm = torch.sqrt(torch.clamp(d_norm_sq, min=1e-12))
sigma_hat = torch.sqrt(torch.abs(d_norm_sq - 1.0)) * torch.abs(pred_R.squeeze(-1))
```
**What this does:** Measures the predicted direction's squared norm, calculates its absolute deviation from 1, takes the square root, and scales it by the predicted magnitude to form $\hat{\sigma}_{raw}$.  
**Why:** The deviation from the unit hypersphere ($d\_norm\_sq \neq 1$) is the geometric proxy for the model's epistemic uncertainty.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

We demonstrate HCM on a toy 1D regression dataset.
Let $n=4$ calibration samples and $m=3$ test samples. We aim to calibrate raw uncertainties using a grid search over $s \in \{0.8, 1.0, 1.2\}$ to match target $1\sigma$ coverage (68%).

**Calibration Set:**
| Sample $i$ | True Target $y_i$ |
|---|---|
| 1 | 2.5 |
| 2 | 5.0 |
| 3 | 7.5 |
| 4 | 10.0 |

**Test Set Design:**
- **Easy Sample:** Low noise, prediction is highly confident.
- **Borderline Sample:** Moderate noise.
- **Ambiguous Sample:** Out-of-distribution or highly noisy, low confidence.

### 5.1 Method: HCM

#### Step A — Compute Raw Uncertainty on the Calibration Set

For each calibration sample, the model predicts $\hat{R}$ and $\hat{\mathbf{d}}$.

| $i$ | True $y_i$ | Pred $\hat{y}_i$ | Abs Error $e_i$ | Pred $\hat{R}$ | Pred $||\hat{\mathbf{d}}||^2$ | Raw $\hat{\sigma}_{raw} = \sqrt{| ||\hat{\mathbf{d}}||^2 - 1 |} \cdot \hat{R}$ |
|---|---|---|---|---|---|---|
| 1 | 2.5 | 2.3 | 0.2 | 3.25 | 0.98 | $\sqrt{|0.98 - 1|} \cdot 3.25 = 0.46$ |
| 2 | 5.0 | 5.4 | 0.4 | 7.63 | 0.96 | $\sqrt{|0.96 - 1|} \cdot 7.63 = 1.53$ |
| 3 | 7.5 | 6.5 | 1.0 | 9.19 | 0.91 | $\sqrt{|0.91 - 1|} \cdot 9.19 = 2.76$ |
| 4 | 10.0| 10.3| 0.3 | 14.56| 0.99 | $\sqrt{|0.99 - 1|} \cdot 14.56 = 1.45$ |

*Absolute Errors:* $[0.2, 0.4, 1.0, 0.3]$
*Raw $\hat{\sigma}_{raw}$:* $[0.46, 1.53, 2.76, 1.45]$

#### Step B — Compute the Temperature Scale

We test values for $s$ to see which provides the best 68% coverage (i.e. $1\sigma$ interval covers $y_i$). We need $\approx 68\%$ of samples (3 out of 4) to satisfy $e_i \le s \cdot \hat{\sigma}_{raw}$.

- **If $s = 0.8$:**
  - Scaled $\hat{\sigma}$: $[0.37, 1.22, 2.21, 1.16]$
  - Covered? ($e_i \le \hat{\sigma}$): [False ($0.2 \le 0.37 \rightarrow$ True), True ($0.4 \le 1.22$), True ($1.0 \le 2.21$), True ($0.3 \le 1.16$)] -> 100% Coverage
- **If $s = 0.5$:**
  - Scaled $\hat{\sigma}$: $[0.23, 0.76, 1.38, 0.72]$
  - Covered? [True, True, True, True] -> 100% Coverage

*(Note: Real grid search balances $1\sigma, 2\sigma, 3\sigma$ coverage simultaneously via MSE. Let's assume the optimal scale chosen by the search is $\hat{s} = 0.8$.)*

#### Step C — Build Prediction Intervals for Test Samples

Applying the optimal scale $\hat{s} = 0.8$ to our test samples:

| Test Sample | True $y$ | Pred $\hat{y}$ | $||\hat{\mathbf{d}}||^2$ | $\hat{R}$ | Raw $\hat{\sigma}$ | Scaled $\hat{\sigma}_{calibrated}$ | 95% Interval ($\hat{y} \pm 1.96\hat{\sigma}$) | Covered? |
|---|---|---|---|---|---|---|---|---|
| **Easy** | 4.0 | 4.1 | 0.99 | 5.8 | $0.58$ | $0.46$ | $[3.20, 5.00]$ | ✓ |
| **Borderline**| 8.0 | 6.8 | 0.94 | 9.6 | $2.35$ | $1.88$ | $[3.11, 10.48]$ | ✓ |
| **Ambiguous** | 15.0| 11.5| 0.81 | 16.2| $7.06$ | $5.65$ | $[0.43, 22.57]$ | ✓ |

**What this means:**
- For the **easy sample**, the predicted direction is very close to the unit sphere ($||\hat{\mathbf{d}}||^2 = 0.99$), resulting in a tight prediction interval.
- For the **ambiguous sample**, the model is highly uncertain, causing a massive norm collapse ($||\hat{\mathbf{d}}||^2 = 0.81$) which balloons the predicted standard deviation to $5.65$, expanding the interval enough to safely cover the large absolute error.

#### Step D — Summary Table

| Test Sample | Prediction $\hat{y}$ | Standard Deviation $\hat{\sigma}$ | 95% Interval | Coverage |
|---|---|---|---|---|
| Easy | 4.10 | 0.46 | $[3.20, 5.00]$ | ✓ |
| Borderline | 6.80 | 1.88 | $[3.11, 10.48]$ | ✓ |
| Ambiguous | 11.50 | 5.65 | $[0.43, 22.57]$ | ✓ |

---

## 6. References

[1] Ryu, S., et al. "Hyperspherical Confidence Mapping (HCM)." International Conference on Learning Representations (ICLR), 2026. [OpenReview Link](https://openreview.net/forum)  
[2] "Hyperspherical Confidence Mapping Code Repository." GitHub. [Link](https://github.com/DaneshSelwal/treeffuser) *(Inferred from notebook implementation)*

# Classification and Regression Diffusion Models (CARD): Theory & Implementation Summary

> **One-line description:** CARD recovers the full conditional distribution of a response variable by applying a denoising diffusion process to the residuals of a pre-trained conditional mean estimator.

---

## 1. Overview & Intuition

Traditional supervised learning models excel at predicting the conditional mean $\mathbb{E}[y|x]$ of a target variable $y$ given covariates $x$. However, in many real-world applications (such as climate forecasting, finance, or medical diagnosis), point estimates are insufficient. Decision-makers need to understand the full predictive uncertainty and potential multi-modality of the data distribution $P(y|x)$. 

Classification and Regression Diffusion Models (CARD) address this by adapting the powerful framework of Denoising Diffusion Probabilistic Models (DDPMs) to supervised tasks. Instead of directly generating $y$ from noise, CARD leverages a pre-trained base regression model to provide a strong conditional mean estimate. It then models the *residual* uncertainty—the difference between the true $y$ and the predicted mean—using a forward noise-addition process and a learned reverse denoising process conditioned on $x$.

The core insight behind CARD is that the structural relationship between $x$ and $y$ can be largely captured by standard regression techniques (like Random Forests or Gradient Boosting), while the complex, potentially non-Gaussian residual distribution can be modeled iteratively via diffusion. This hybrid approach ensures high fidelity in conditional generation and robust uncertainty quantification, offering a full probabilistic distribution rather than just a point prediction.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X}$ be the input feature space and $\mathcal{Y}$ be the continuous output space for regression. Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, our goal is to learn the conditional distribution $p(y|x)$. We assume access to a pre-trained base model $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ that estimates the conditional mean $\hat{y} = f_\theta(x)$.

### 2.2 Residual Formulation

Instead of applying diffusion to $y$ directly, CARD applies it to the residual $r$:

**Equation:**
$$r = y - f_\theta(x)$$

**Where:**
- $r$ — the residual (the unexplained portion of the target)
- $y$ — the true target value
- $f_\theta(x)$ — the predicted mean from the base model

**What this means:** By shifting the target to the residual, the diffusion model only needs to learn the noise distribution around the mean, which is typically zero-centered and easier to model conditionally.

### 2.3 Forward Diffusion Process

The forward process gradually adds Gaussian noise to the residual $r$ over $T$ timesteps, producing a sequence of noisy variables $r_0, r_1, \dots, r_T$, where $r_0 = r$.

**Equation:**
$$q(r_t | r_{t-1}) = \mathcal{N}(r_t; \sqrt{1 - \beta_t} r_{t-1}, \beta_t \mathbf{I})$$

**Where:**
- $r_t$ — the noisy residual at timestep $t$
- $\beta_t$ — the variance schedule parameter at step $t$
- $\mathcal{N}$ — the Gaussian distribution

**What this means:** At each step, a small amount of variance $\beta_t$ is injected. By the final step $T$, $r_T$ is nearly indistinguishable from pure isotropic Gaussian noise $\mathcal{N}(0, \mathbf{I})$.

### 2.4 Reverse Denoising Process

The reverse process learns to reconstruct the original residual $r_0$ from the noise $r_T$, conditioned on the input $x$. A neural network $\epsilon_\phi(x, t)$ is trained to predict the added noise at step $t$.

**Equation:**
$$\mathcal{L}_{CARD} = \mathbb{E}_{x, r_0, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\phi(x, t) \|^2 \right]$$

**Where:**
- $\mathcal{L}_{CARD}$ — the training objective (Mean Squared Error)
- $\epsilon$ — the actual Gaussian noise added to $r_0$ to get $r_t$
- $\epsilon_\phi(x, t)$ — the predicted noise from the neural network (MLP)
- $t$ — the randomly sampled timestep

**What this means:** The neural network learns to predict the exact noise vector $\epsilon$ that was injected at timestep $t$. Once trained, we can sample $r_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively denoise it to generate realistic samples of $r_0$.

---

## 3. Algorithm

**Input:** Training data $(X, Y)$, Base Model $f_\theta$, Timesteps $T$, Noise MLP $\epsilon_\phi$  
**Output:** Predictive distribution parameters (Mean and StdDev) for test points $X_{test}$  

1. **Base Training:** Train the base regressor $f_\theta$ on $(X, Y)$.
2. **Residual Calculation:** Compute the residuals $R = Y - f_\theta(X)$.
3. **Diffusion Training:** For $epochs$:
   1. Sample timestep $t \sim \text{Uniform}(1, T)$.
   2. Sample noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.
   3. Add noise to $R$ to get $R_t$. *(Note: The simplified notebook variant predicts noise directly without the full sequence schedule).*
   4. Predict noise $\hat{\epsilon} = \epsilon_\phi(X, t)$.
   5. Update $\epsilon_\phi$ using gradient descent on $\| \epsilon - \hat{\epsilon} \|^2$.
4. **Inference (Sampling):** For a test point $x_{test}$:
   1. Compute base prediction $\hat{y} = f_\theta(x_{test})$.
   2. Sample pure noise $r_T \sim \mathcal{N}(0, \mathbf{I})$.
   3. Denoise $r_T$ to $r_0$ using $\epsilon_\phi(x_{test}, t)$.
   4. Final sample $y^{(s)} = \hat{y} + r_0$.
   5. Repeat $N$ times to form an empirical distribution.
   6. Compute empirical Mean and StdDev of the samples.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Probabilistic__Distribution(CARD).ipynb`

The notebook implements a **highly simplified variant** of the CARD method. Instead of simulating the full forward SDE and reverse step-by-step Markov chain, it uses an MLP to predict noise in a single step, serving as an empirical approximation of residual distributions.

### 4.1 Initialization and Architecture
```python
class CARDRegressor:
    def __init__(self, base_model, hidden_dim=128, T=50, n_samples=100, ...):
        self.base_model = base_model
        self.T = T
        self.n_samples = n_samples
        self.loss_fn = torch.nn.MSELoss()
```
**What this does:** Initializes the `CARDRegressor` wrapper, which accepts any pre-trained scikit-learn compatible `base_model` (e.g., Random Forest, XGBoost).  
**Why:** This allows the method to decouple the conditional mean estimation (handled by `base_model`) from the uncertainty quantification (handled by the diffusion process).

### 4.2 Training the Noise Predictor (Simplified Diffusion)
```python
    def fit(self, X, y):
        y_pred = self.base_model.predict(X)
        residuals = y - y_pred
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1] + 1, self.hidden_dim),
            ...
            torch.nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        for _ in range(self.epochs):
            t = torch.randint(0, self.T, (len(X_t),), dtype=torch.float32)
            noise = torch.randn_like(r_t)

            pred_noise = self.mlp(torch.cat([X_t, t.unsqueeze(1)], dim=1)).squeeze()
            loss = self.loss_fn(pred_noise, noise)
            ...
```
**What this does:** Computes the residuals $r = y - \hat{y}$. It then trains an MLP that takes the concatenated features $X$ and a randomized timestep $t$, and attempts to predict randomly generated standard normal `noise`.  
**Why:** In standard CARD, the MLP predicts the noise added to $r_t$. Here, the simplified notebook version trains the MLP to map $(X, t)$ directly to a noise space, learning the variance scaling implicitly over epochs.

### 4.3 Inference and Distribution Sampling
```python
    def predict(self, X):
        base_pred = self.base_model.predict(X)
        samples = []
        for _ in range(self.n_samples):
            t = torch.zeros(len(X_t))
            noise = torch.randn(len(X_t))

            pred_noise = self.mlp(torch.cat([X_t, t.unsqueeze(1)], dim=1)).detach().squeeze()
            samples.append(base_pred + (noise - pred_noise).numpy())

        samples = np.stack(samples, axis=1)
        return samples.mean(axis=1), samples.std(axis=1)
```
**What this does:** Generates `n_samples` predictions for each input $X$. It extracts the base prediction, queries the MLP at $t=0$ with randomly sampled `noise`, and creates a residual sample `(noise - pred_noise)`. It aggregates the $N$ samples to return the empirical Mean and Standard Deviation.  
**Why:** By repeatedly sampling, we build a full conditional predictive distribution $P(y|x)$. The standard deviation naturally forms the uncertainty band (e.g., for constructing $1.96 \times \text{StdDev}$ confidence intervals as seen in the notebook's plotting functions).

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

We adapt the toy dataset for regression to demonstrate how the simplified CARD process estimates uncertainty and forms prediction intervals.

**Toy Dataset:**
- Input dimension: 1 ($X \in \mathbb{R}$)
- Base model $f_\theta$ is pre-trained.
- $N_{samples} = 3$ generated during inference.
- Uncertainty threshold: 95% Confidence Interval ($\mu \pm 1.96 \sigma$).

**Test Samples (3 distinct profiles):**
- **Sample 1 (Easy):** $x = 1.0$. The model has seen lots of similar data. Uncertainty should be low.
- **Sample 2 (Borderline):** $x = 4.5$. At the edge of the training distribution. Moderate uncertainty.
- **Sample 3 (Ambiguous):** $x = 8.0$. Out-of-distribution (OOD). High uncertainty.

| Sample Index | $X$ | True $y$ | Base Pred $\hat{y}$ |
|---|---|---|---|
| 1 | 1.0 | 2.1 | 2.0 |
| 2 | 4.5 | 5.8 | 6.0 |
| 3 | 8.0 | 9.5 | 8.5 |

---

### 5.1 Method: CARD (Simplified Residual Diffusion)

#### Step A — Compute residuals on the calibration/training set
Assume our training set had samples near $x=1.0$ with very small residuals ($r \approx 0.1$), and samples near $x=4.5$ with larger residuals ($r \approx -0.2$). The MLP learns to output predicted noise $\hat{\epsilon}$ based on $X$ and $t$.

#### Step B — Generate multiple samples for Test Points at $t=0$

For each test point, we draw 3 standard normal noise values $\epsilon_i \sim \mathcal{N}(0, 1)$, predict the noise $\hat{\epsilon}_i = \text{MLP}(X, t=0)$, and compute the final sample $y_i = \hat{y} + (\epsilon_i - \hat{\epsilon}_i)$.

**Test Sample 1 (Easy, $X=1.0$, $\hat{y}=2.0$):**
Because $X=1.0$ is well-represented, the MLP predicts the noise accurately ($\hat{\epsilon}_i \approx \epsilon_i$), making the residual difference small.
- Draw 1: $\epsilon_1 = 0.5 \rightarrow \hat{\epsilon}_1 = 0.45 \rightarrow y_1 = 2.0 + (0.5 - 0.45) = 2.05$
- Draw 2: $\epsilon_2 = -1.2 \rightarrow \hat{\epsilon}_2 = -1.15 \rightarrow y_2 = 2.0 + (-1.2 - (-1.15)) = 1.95$
- Draw 3: $\epsilon_3 = 0.8 \rightarrow \hat{\epsilon}_3 = 0.75 \rightarrow y_3 = 2.0 + (0.8 - 0.75) = 2.05$

**Test Sample 2 (Borderline, $X=4.5$, $\hat{y}=6.0$):**
The MLP is less accurate here.
- Draw 1: $\epsilon_1 = 1.0 \rightarrow \hat{\epsilon}_1 = 0.7 \rightarrow y_1 = 6.0 + 0.3 = 6.30$
- Draw 2: $\epsilon_2 = -0.8 \rightarrow \hat{\epsilon}_2 = -0.5 \rightarrow y_2 = 6.0 - 0.3 = 5.70$
- Draw 3: $\epsilon_3 = 0.2 \rightarrow \hat{\epsilon}_3 = -0.1 \rightarrow y_3 = 6.0 + 0.3 = 6.30$

**Test Sample 3 (Ambiguous/OOD, $X=8.0$, $\hat{y}=8.5$):**
The MLP is highly inaccurate out-of-distribution, leading to large stochastic variations.
- Draw 1: $\epsilon_1 = 1.5 \rightarrow \hat{\epsilon}_1 = -0.5 \rightarrow y_1 = 8.5 + 2.0 = 10.5$
- Draw 2: $\epsilon_2 = -1.5 \rightarrow \hat{\epsilon}_2 = 1.0 \rightarrow y_2 = 8.5 - 2.5 = 6.0$
- Draw 3: $\epsilon_3 = 0.0 \rightarrow \hat{\epsilon}_3 = -1.0 \rightarrow y_3 = 8.5 + 1.0 = 9.5$

#### Step C — Compute Predictive Statistics and Confidence Intervals

For each sample, compute Empirical Mean ($\mu_{emp}$), Standard Deviation ($\sigma_{emp}$), and the 95% Confidence Interval ($\mu_{emp} \pm 1.96 \times \sigma_{emp}$).

| Sample | Draws $[y_1, y_2, y_3]$ | $\mu_{emp}$ | $\sigma_{emp}$ | 95% Interval ($\mu_{emp} \pm 1.96 \sigma_{emp}$) | True $y$ | Coverage (In Set?) |
|---|---|---|---|---|---|---|
| 1 | [2.05, 1.95, 2.05] | 2.017 | 0.047 | [1.92, 2.11] | 2.1 | ✓ |
| 2 | [6.30, 5.70, 6.30] | 6.100 | 0.283 | [5.55, 6.65] | 5.8 | ✓ |
| 3 | [10.5, 6.0, 9.5] | 8.667 | 1.925 | [4.89, 12.44] | 9.5 | ✓ |

#### Step D — Summary of Results

| Test Sample | True $y$ | Predicted Mean | Interval Size | Coverage |
|---|---|---|---|---|
| Easy ($X=1.0$) | 2.1 | 2.017 | 0.19 | ✓ |
| Borderline ($X=4.5$) | 5.8 | 6.100 | 1.10 | ✓ |
| Ambiguous ($X=8.0$) | 9.5 | 8.667 | 7.55 | ✓ |

**Analysis:**
The diffusion method intrinsically captures epistemic uncertainty. For the "Easy" sample, the noise predictor is well-calibrated, resulting in a tight prediction interval (size 0.19). For the "Ambiguous" OOD sample, the MLP's failure to predict the noise $\epsilon$ causes the raw noise to dominate the residual, organically expanding the interval size (to 7.55) to safely cover the unknown true value.

---

## 6. References

[1] Han, X., Zheng, J., & Zhou, M. "CARD: Classification and Regression Diffusion Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [arXiv:2206.07275](https://arxiv.org/abs/2206.07275)

[2] Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

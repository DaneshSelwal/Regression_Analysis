# Optuna Hyperparameter Optimization and Explainability: Theory & Implementation Summary

> **One-line description:** An automated hyperparameter tuning framework using Optuna's Tree-structured Parzen Estimator (TPE) combined with LIME and SHAP for model interpretability in regression tasks.

---

## 1. Overview & Intuition

Hyperparameter tuning is crucial for maximizing the performance of complex machine learning models, especially ensemble regression methods like XGBoost, LightGBM, and NGBoost. Traditional methods like Grid Search are computationally prohibitive, while Random Search is inefficient. Optuna introduces an efficient Bayesian optimization framework, using the Tree-structured Parzen Estimator (TPE), to iteratively sample promising hyperparameter regions based on prior evaluations.

Furthermore, highly accurate ensemble models are often "black boxes." To establish trust in their predictions—especially in critical domains like hydrology (predicting future states from past observations)—we must explain how features influence the output. This notebook pairs the optimized models with Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) to provide both local and global feature importance metrics, ensuring that the selected models are both performant and transparent.

---

## 2. Mathematical Framework

### 2.1 Tree-structured Parzen Estimator (TPE)

Optuna's default optimization algorithm is TPE. Instead of modeling the probability of the objective function given the hyperparameters $P(y|x)$ (like standard Gaussian Processes), TPE models $P(x|y)$ and $P(y)$.

**Equation:**
$$ P(x|y) = \begin{cases} l(x) & \text{if } y < y^* \\ g(x) & \text{if } y \ge y^* \end{cases} $$

**Where:**
- $x$ — hyperparameter configuration
- $y$ — objective function value (e.g., RMSE)
- $y^*$ — threshold value of the objective function (e.g., the $\gamma$-quantile of observed losses)
- $l(x)$ — density formed by configurations associated with losses lower than $y^*$
- $g(x)$ — density formed by configurations associated with losses greater than or equal to $y^*$

**What this means:** TPE divides previous trials into "good" and "bad" groups and builds two separate density distributions. New hyperparameters are sampled to maximize the ratio $l(x)/g(x)$, encouraging exploration in regions where the "good" density is high and "bad" density is low.

### 2.2 SHapley Additive exPlanations (SHAP)

SHAP values allocate the difference between a model's prediction for a specific instance and the global average prediction among the input features, using coalitional game theory.

**Equation:**
$$ \phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right] $$

**Where:**
- $\phi_i(f, x)$ — SHAP value (contribution) of feature $i$ for instance $x$
- $F$ — set of all features
- $S$ — subset of features not including $i$
- $f_x(S)$ — the expected model output conditioned on the features in subset $S$

**What this means:** The importance of feature $i$ is calculated by observing how the prediction changes when feature $i$ is added to every possible subset of other features, weighted by the number of permutations.

### 2.3 Local Interpretable Model-agnostic Explanations (LIME)

LIME explains a prediction by training a simple, interpretable surrogate model (e.g., linear regression) on locally perturbed samples around the instance of interest.

**Equation:**
$$ \text{explanation}(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

**Where:**
- $x$ — instance being explained
- $g$ — interpretable surrogate model from a class of models $G$ (e.g., linear models)
- $f$ — original complex model
- $\pi_x$ — proximity measure (weight) for perturbed samples around $x$
- $\mathcal{L}$ — loss function measuring how unfaithful $g$ is in approximating $f$ locally
- $\Omega(g)$ — complexity penalty for the surrogate model

**What this means:** LIME generates synthetic data points around $x$, queries the complex model $f$ for their predictions, and fits a simple weighted model $g$ that approximates $f$ well in that local neighborhood while remaining simple (low $\Omega(g)$).

---

## 3. Algorithm

**Input:** Training data $X_{\text{train}}, y_{\text{train}}$, testing data $X_{\text{test}}, y_{\text{test}}$, defined hyperparameter search spaces for each model, number of Optuna trials.  
**Output:** Optimized models, best hyperparameters, RMSE and Correlation plots, LIME & SHAP feature importance scores.

1. Normalize features using Standard Normalization (z-score).
2. For each defined regression model (e.g., RandomForest, XGBoost, GPBoost):
3. Define an Optuna objective function that instantiates the model with sampled hyperparameters.
4. Apply Cross-Validation or a hold-out set to evaluate the objective (RMSE).
5. Apply Optuna Pruner (e.g., MedianPruner) to halt unpromising trials early.
6. Record the best hyperparameters and retrain the model on the full training set.
7. Evaluate test set RMSE and Correlation Coefficient.
8. Compute SHAP values via `ShapKernel` and LIME feature importances via `LimeTabular` for the test instances.
9. Aggregate the local explanations to compute the mean absolute importance for each feature.
10. Save the metrics and plots to an Excel summary file.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Optuna_autosampler.ipynb`

### 4.1 Data Preparation and Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
**What this does:** Applies z-score normalization to ensure all features have zero mean and unit variance.  
**Why:** Many machine learning models (like TabNet, and distance-based methods) are sensitive to the scale of input features. Normalization ensures stable and faster convergence.

### 4.2 Optuna Trial Execution
```python
# Extracted from standard Optuna workflow in the notebook
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30)
    }
    model = RandomForestRegressor(**params)
    ...
```
**What this does:** Uses Optuna's trial object to sample hyperparameters from predefined distributions for each model.  
**Why:** This allows the TPE sampler to actively propose new hyperparameters based on previous evaluations, optimizing the search process.

### 4.3 Feature Importance Aggregation
```python
lime_explainer = LimeTabular(predict_fn, data=data_for_explainer, ...)
lime_explanation = lime_explainer.explain_local(data_for_explanation)

for idx in range(num_instances):
    explanation = lime_explanation.data(idx)
    for feature_name, feature_score in zip(explanation['names'], explanation['scores']):
        feature_importances_lime[feature_name] += abs(feature_score)
```
**What this does:** Generates LIME local explanations for a set of test instances and accumulates the absolute feature scores across all instances.  
**Why:** To transition from local instance-based interpretability to a global understanding of feature importance for the specific model.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

Since this is a regression problem (predicting continuous storage $S_t$), we adapt the toy example rules to demonstrate regression optimization and feature attribution.

**Toy Dataset Setup:**
We define a toy dataset with $n=5$ training samples and $3$ test samples, predicting a target $y$ from features $x_1, x_2$.
Target function: $y = 2x_1 + 0.5x_2$

**Calibration / Training Data:**

| Sample | $x_1$ | $x_2$ | True $y$ |
|---|---|---|---|
| 1 | 1.0 | 2.0 | 3.0 |
| 2 | 2.0 | 1.0 | 4.5 |
| 3 | 3.0 | 2.0 | 7.0 |
| 4 | 4.0 | 3.0 | 9.5 |
| 5 | 5.0 | 1.0 | 10.5 |

**Test Data (Exposing Method Behaviors):**
- **Test A (Easy):** $x_1=2.0, x_2=2.0$ (True $y = 5.0$)
- **Test B (Out of distribution):** $x_1=8.0, x_2=1.0$ (True $y = 16.5$)
- **Test C (Noisy):** $x_1=1.0, x_2=8.0$ (True $y = 6.0$)

---

### 5.1 Method 1: SHAP Feature Attribution

Assume our tuned model $f(x)$ perfectly learned the function $f(x) = 2x_1 + 0.5x_2$. We want to compute the SHAP values for **Test A**.

#### Step A — Compute the Base Value
The base value $E[f(x)]$ is the average prediction over the training set:
$$E[f(x)] = \frac{3.0 + 4.5 + 7.0 + 9.5 + 10.5}{5} = \frac{34.5}{5} = 6.9$$

#### Step B — Compute Marginal Contributions for Test A
For Test A ($x_1=2.0, x_2=2.0$), the model predicts $f(Test A) = 2(2.0) + 0.5(2.0) = 5.0$.

SHAP requires evaluating all feature subsets:
- $S = \emptyset$: $f(\emptyset) = 6.9$
- $S = \{x_1\}$: $E[f(2.0, x_2)]$. The expected value of $0.5x_2$ over training data is $0.5 \times 1.8 = 0.9$. So $f(\{x_1\}) = 2(2.0) + 0.9 = 4.9$.
- $S = \{x_2\}$: $E[f(x_1, 2.0)]$. The expected value of $2x_1$ over training data is $2 \times 3.0 = 6.0$. So $f(\{x_2\}) = 6.0 + 0.5(2.0) = 7.0$.
- $S = \{x_1, x_2\}$: $f(\{x_1, x_2\}) = 5.0$.

#### Step C — Calculate SHAP Values
For $x_1$:
- When added to $\emptyset$: $4.9 - 6.9 = -2.0$ (weight $1/2$)
- When added to $\{x_2\}$: $5.0 - 7.0 = -2.0$ (weight $1/2$)
$\phi_1 = 0.5(-2.0) + 0.5(-2.0) = -2.0$

For $x_2$:
- When added to $\emptyset$: $7.0 - 6.9 = 0.1$ (weight $1/2$)
- When added to $\{x_1\}$: $5.0 - 4.9 = 0.1$ (weight $1/2$)
$\phi_2 = 0.5(0.1) + 0.5(0.1) = 0.1$

Check: $E[f(x)] + \phi_1 + \phi_2 = 6.9 - 2.0 + 0.1 = 5.0$, which perfectly matches the model output.

#### Step D — Summary Table for Test Instances

| Test Sample | $x_1$ | $x_2$ | Prediction | $\phi_1$ | $\phi_2$ | Base Value |
|---|---|---|---|---|---|---|
| Test A | 2.0 | 2.0 | 5.0 | -2.0 | +0.1 | 6.9 |
| Test B | 8.0 | 1.0 | 16.5 | +10.0 | -0.4 | 6.9 |
| Test C | 1.0 | 8.0 | 6.0 | -4.0 | +3.1 | 6.9 |

---

### 5.2 Optuna Optimization vs Grid Search (Cross-Method Comparison)

If we optimized a single parameter $\alpha \in [0.1, 10]$ where true optimum is $\alpha=5.5$:

| Iteration | Grid Search $\alpha$ | TPE Sampler $\alpha$ |
|---|---|---|
| 1 | 0.1 | 2.3 |
| 2 | 2.5 | 8.1 |
| 3 | 5.0 | 5.0 |
| 4 | 7.5 | 5.8 |
| 5 | 10.0 | 5.4 |

**Why the methods differ:**
Grid Search blindly evaluates predefined points, entirely missing the exact minimum. The TPE Sampler iteratively narrows its search distribution (the $l(x)$ density) around the best observed regions (e.g., between 5.0 and 5.8), converging rapidly to the optimal $\alpha = 5.4$ by iteration 5.

---

## 6. References

[1] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD, 2019. [https://arxiv.org/abs/1907.10902](https://arxiv.org/abs/1907.10902)

[2] Ribeiro, M. T., Singh, S., & Guestrin, C. ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." KDD, 2016. [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)

[3] Lundberg, S. M., & Lee, S. I. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017. [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)

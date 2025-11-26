# Bayesian Inference Reimagined for R, Python and Julia

<div align="center">

**A unified probabilistic programming library bridging the gap between user-friendly R syntax and high-performance JAX computation.**  
*Run bespoke models on CPU, GPU, or TPU with ease.*

[![License: GPL (>= 3)](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R build status](https://github.com/BGN-for-ASNA/BIR/workflows/R-CMD-check/badge.svg)](https://github.com/BGN-for-ASNA/BIR/actions)

</div>

---

## One Mental Model. Three Languages.

Whether you prefer R's formula syntax or Python's object-oriented approach, **BayesianInference (BI)** unifies the experience.

-   ✅ **Zero Context Switching**: Variable names, distribution signatures, and model logic remain consistent.
-   ✅ **NumPyro Power**: Both interfaces compile down to XLA via JAX for blazing fast inference.
-   ✅ **Rich Diagnostics**: Seamless integration with ArviZ for posterior analysis.

### Compare the Syntax

<table width="100%">
<tr>
<th width="33%">R Syntax</th>
<th width="33%">Python Syntax</th>
<th width="33%">Julia Syntax</th>
</tr>
<tr>
<td valign="top">

```r
model <- function(height, weight){
  # Priors
  sigma = bi.dist.uniform(0, 50, name='sigma', shape=c(1))
  alpha = bi.dist.normal(178, 20, name='alpha', shape=c(1))
  beta  = bi.dist.normal(0, 1, name='beta', shape=c(1))

  # Likelihood
  mu = alpha + beta * weight
  bi.dist.normal(mu, sigma, obs=height)
}
```

</td>
<td valign="top">

```python
def model(height, weight):
    # Priors
    sigma = bi.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bi.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bi.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bi.dist.normal(mu, sigma, obs=height)
```

<td valign="top">

```Julia
@BI function model(weight, height)
    # Priors
    sigma = bi.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bi.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bi.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bi.dist.normal(mu, sigma, obs=height)
end
```

</details>

</td>
</tr>
</table>

---

## Built for Speed

Leveraging Just-In-Time (JIT) compilation, BI outperforms traditional engines on standard hardware and unlocks massive scalability on GPU clusters for large datasets.

**Benchmark: Network Size 100 (Lower is Better)**

| Engine | Execution Time | Relative Performance |
| :--- | :--- | :--- |
| **STAN (CPU)** | `████████████████████████████` | *Baseline* |
| **BI (CPU)** | `████████████` | **~2.5x Faster** |

*> Comparison of execution time for a Social Relations Model. Source: Sosa et al. (2025).*

---

## Installation & Setup

### 1. Install Python
Download and install [Python](https://www.python.org/downloads/)

### 2. Install Package
Use `devtools` to pull the latest development version from GitHub.

```r
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
devtools::install_github("https://github.com/BGN-for-ASNA/BIR")
```

### 3. Initialize Environment
Run the starting test to create the Python virtual environment managed by `reticulate`.

```r
library(BayesianInference)
# Run the starting test to install Python dependencies
BI_starting_test()
```

### 4. Select Backend
Choose `'cpu'`, `'gpu'`, or `'tpu'` when importing the library.

```r
# Initialize on CPU (default) or GPU/TPU
m <- importBI(platform = 'cpu')
```

---

## Features

### Data Manipulation
-   One-hot encoding
-   Index variable conversion
-   Scaling and normalization

### Modeling (via NumPyro)
-   **Linear & Generalized Linear Models**: Regression, Binomial, Poisson, Negative Binomial, etc.
-   **Hierarchical/Multilevel Models**: Varying intercepts and slopes.
-   **Time Series & Processes**: Gaussian Processes, Gaussian Random Walks, State Space Models.
-   **Mixture Models**: GMM, Dirichlet Process Mixtures.
-   **Network Models**: Network-based diffusion, Block models.
-   **Bayesian Neural Networks (BNN)**.

### Diagnostics (via ArviZ)
-   Posterior summary statistics and plots.
-   Trace plots, Density plots, Autocorrelation.
-   WAIC and LOO (ELPD) model comparison.
-   R-hat and Effective Sample Size (ESS).

---

## Available Distributions

The package provides wrappers for a comprehensive set of distributions from NumPyro.

### Continuous
-   `bi.dist.normal`, `bi.dist.uniform`, `bi.dist.student_t`
-   `bi.dist.cauchy`, `bi.dist.halfcauchy`, `bi.dist.halfnormal`
-   `bi.dist.gamma`, `bi.dist.inverse_gamma`, `bi.dist.exponential`
-   `bi.dist.beta`, `bi.dist.beta_proportion`
-   `bi.dist.laplace`, `bi.dist.asymmetric_laplace`
-   `bi.dist.log_normal`, `bi.dist.log_uniform`
-   `bi.dist.pareto`, `bi.dist.weibull`, `bi.dist.gumbel`
-   `bi.dist.chi2`, `bi.dist.gompertz`

### Discrete
-   `bi.dist.bernoulli`, `bi.dist.binomial`
-   `bi.dist.poisson`, `bi.dist.negative_binomial`
-   `bi.dist.geometric`, `bi.dist.discrete_uniform`
-   `bi.dist.beta_binomial`, `bi.dist.zero_inflated_poisson`

### Multivariate
-   `bi.dist.multivariate_normal`, `bi.dist.multivariate_student_t`
-   `bi.dist.dirichlet`, `bi.dist.dirichlet_multinomial`
-   `bi.dist.multinomial`
-   `bi.dist.lkj`, `bi.dist.lkj_cholesky`
-   `bi.dist.wishart`, `bi.dist.wishart_cholesky`

### Time Series & Stochastic Processes
-   `bi.dist.gaussian_random_walk`
-   `bi.dist.gaussian_state_space`
-   `bi.dist.euler_maruyama`
-   `bi.dist.car` (Conditional AutoRegressive)

### Mixtures & Truncated
-   `bi.dist.mixture`, `bi.dist.mixture_same_family`
-   `bi.dist.truncated_normal`, `bi.dist.truncated_cauchy`
-   `bi.dist.lower_truncated_power_law`

*(See package documentation for the full list)*

---

## Documentation

For full documentation of functions and parameters, you can use the built-in R help or the package helper:

```r
# Open package documentation
bi.doc()

# Help for a specific function
?bi.dist.normal
```
## Platform Support

-   ✅ Linux
-   ✅ macOS
-   ✅ Windows

GPU support available on compatible systems with JAX GPU installation.

---

## Related Packages

-   [BI](https://pypi.org/project/BayesInference) - Python implementation
-   [BIJ](https://github.com/BGN-for-ASNA/BIJ) - J implementation

---

<div align="center">

**BayesianInference (BIR)**  
Based on "The Bayesian Inference library for Python, R and Julia" by Sosa, McElreath, & Ross (2025).

[Documentation](#) | [GitHub](https://github.com/BGN-for-ASNA/BIR) | [Issues](https://github.com/BGN-for-ASNA/BIR/issues)

&copy; 2025 BayesianInference Team. Released under GPL-3.0.

</div>

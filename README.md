# Bayesian Inference Reimagined for R, Python and Julia

<div align="center">

**A unified probabilistic programming library bridging the gap between user-friendly R syntax and high-performance JAX computation.**  
*Run bespoke models on CPU, GPU, or TPU with ease.*

[![Website](https://img.shields.io/badge/Website-s--sosa.com/BI-blue?style=flat&logo=google-chrome&logoColor=white)](https://s-sosa.com/BI/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.01.19.700318v1-BD271E?style=flat&logo=biorxiv&logoColor=white)](https://www.bforxiv.org/content/10.64898/2026.01.19.700318v1) 
![R](https://img.shields.io/badge/R-4.0+-276DC3?logo=r&logoColor=white)
[![License: GPL (>= 3)](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

## One Mental Model. Three Languages.

**BFR (BI)** provides a unified experience across R, Python, and Julia. Whether you work in R's formula syntax, Python's object-oriented approach, or Julia's mathematical elegance, the model logic remains consistent.

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
  sigma = bf.dist.uniform(0, 50, name='sigma', shape=c(1))
  alpha = bf.dist.normal(178, 20, name='alpha', shape=c(1))
  beta  = bf.dist.normal(0, 1, name='beta', shape=c(1))

  # Likelihood
  mu = alpha + beta * weight
  bf.dist.normal(mu, sigma, obs=height)
}
```

</td>
<td valign="top">

```python
def model(height, weight):
    # Priors
    sigma = bf.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bf.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bf.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bf.dist.normal(mu, sigma, obs=height)
```

<td valign="top">

```Julia
@BI function model(weight, height)
    # Priors
    sigma = bf.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bf.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bf.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bf.dist.normal(mu, sigma, obs=height)
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

| Engine         | Execution Time                 | Relative Performance |
|:---------------|:-------------------------------|:---------------------|
| **STAN (CPU)** | `████████████████████████████` | *Baseline*           |
| **BI (CPU)**   | `████████████`                 | **\~30x Faster**     |
| **BI (GPU)**   | `██`                           | **\~200x Faster**    |

*\> Comparison of execution time for a Social Relations Model. Source: [Sosa et al. (2026)](https://www.bforxiv.org/content/10.64898/2026.01.19.700318v1).*


---

## Installation & Setup


### 1. Install Package
Use `devtools` to pull the latest development version from GitHub.

```r
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
devtools::install_github("https://github.com/BGN-for-ASNA/BIR")
```

### 2. Initialize Environment
Run the starting test to create the Python virtual environment managed by `reticulate`.

```r
library(BayesForge)
# Run the starting test to install Python dependencies
BF_starting_test()
```

### 3. Select Backend
Choose `'cpu'`, `'gpu'`, or `'tpu'` when importing the library.

```r
# Initialize on CPU (default) or GPU/TPU
m <- importBF(platform = 'cpu')
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
-   `bf.dist.normal`, `bf.dist.uniform`, `bf.dist.student_t`
-   `bf.dist.cauchy`, `bf.dist.halfcauchy`, `bf.dist.halfnormal`
-   `bf.dist.gamma`, `bf.dist.inverse_gamma`, `bf.dist.exponential`
-   `bf.dist.beta`, `bf.dist.beta_proportion`
-   `bf.dist.laplace`, `bf.dist.asymmetric_laplace`
-   `bf.dist.log_normal`, `bf.dist.log_uniform`
-   `bf.dist.pareto`, `bf.dist.weibull`, `bf.dist.gumbel`
-   `bf.dist.chi2`, `bf.dist.gompertz`

### Discrete
-   `bf.dist.bernoulli`, `bf.dist.bfnomial`
-   `bf.dist.poisson`, `bf.dist.negative_binomial`
-   `bf.dist.geometric`, `bf.dist.discrete_uniform`
-   `bf.dist.beta_binomial`, `bf.dist.zero_inflated_poisson`

### Multivariate
-   `bf.dist.multivariate_normal`, `bf.dist.multivariate_student_t`
-   `bf.dist.dirichlet`, `bf.dist.dirichlet_multinomial`
-   `bf.dist.multinomial`
-   `bf.dist.lkj`, `bf.dist.lkj_cholesky`
-   `bf.dist.wishart`, `bf.dist.wishart_cholesky`

### Time Series & Stochastic Processes
-   `bf.dist.gaussian_random_walk`
-   `bf.dist.gaussian_state_space`
-   `bf.dist.euler_maruyama`
-   `bf.dist.car` (Conditional AutoRegressive)

### Mixtures & Truncated
-   `bf.dist.mixture`, `bf.dist.mixture_same_family`
-   `bf.dist.truncated_normal`, `bf.dist.truncated_cauchy`
-   `bf.dist.lower_truncated_power_law`

*(See package documentation for the full list)*

---

## Documentation

For full documentation of functions and parameters, you can use the built-in R help or the package helper:

```r
# Open package documentation
bf.doc()

# Help for a specific function
?bf.dist.normal
```

## Platform Support

-   ✅ Linux
-   ✅ macOS
-   ✅ Windows

GPU support available on compatible systems with JAX GPU installation.

---

## Related Packages

-   [BI](https://pypi.org/project/BFR) - Python implementation
-   [BIJ](https://github.com/BGN-for-ASNA/pyBFR.jl) - J implementation

---

<div align="center">

**BFR (BIR)**  
Based on "The Bayesian Inference library for Python, R, Julia" by [Sosa, McElreath, & Ross (2026)](https://www.bforxiv.org/content/10.64898/2026.01.19.700318v1).

[Official website](https://s-sosa.com/BI) \| [Issues](https://github.com/BGN-for-ASNA/BIJ/issues) \| [Quick Start](https://s-sosa.com/BI/start/Installation.html)

© 2026 Bayesian Inference Team. Released under GPL-3.0.
</div>

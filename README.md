# BIR
Bayesian Inference for R 
## Installation
```r
devtools::install_github("https://github.com/BGN-for-ASNA/BIR")
```

## Overview

Currently, the package provides:

+ Data manipulation:
    + One-hot encoding
    + Conversion of index variables
    + Scaling
      
+ Models (Using Numpyro):
  
    + Linear Regression for continuous variable
    + Multiple continuous Variable
    + Interaction between variables
    + Categorical variable
    + Binomial model
    + Beta binomial
    + Poisson model
    + Gamma-Poisson
    + Multinomial
    + Dirichlet model
    + Zero inflated
    + Varying intercept
    + Varying slopes
    + Gaussian processes
    + Measuring error
    + Latent variable
    + PCA
    + GMM
    + DPMM
    + Network model
    + Network with block model
    + Network control for data collection biases 
    + Network metrics
    + Network Based Diffusion analysis
    + BNN

+ Model diagnostics (using ARVIZ):
    + Data frame with summary statistics
    + Plot posterior densities
    + Bar plot of the autocorrelation function (ACF) for a sequence of data
    + Plot rank order statistics of chains
    + Forest plot to compare HDI intervals from a number of distributions
    + Compute the widely applicable information criterion
    + Compare models based on their expected log pointwise predictive density (ELPD)
    + Compute estimate of rank normalized split-R-hat for a set of traces
    + Calculate estimate of the effective sample size (ESS)
    + Pair plot
    + Density plot
    + ESS evolution plot
      
# Model and Results Comparisons
This package has been built following the Rethinking Classes of 2024.

# Why?
## 1.  To learn

## 2.  Easy Model Building:
The following linear regression model (rethinking 4.Geocentric Models): 
```math
height∼Normal(μ,σ)
```
```math
μ=α+β*weight
```
```math 
α∼Normal(178,20)
```
```math
β∼Normal(0,10)
```
```math
σ∼Uniform(0,50)
```
    
can be declared in the package as
```
library(BayesianInference)
m=importBI(platform='cpu')

# Load csv file
m$data(paste(system.file(package = "BayesianInference"),"/data/Howell1.csv", sep = ''), sep=';')

# fileter data frame
m$df = m$df[m$df$age > 18,]

# Scale
m$scale(list('weight')) 

# convert data to jax arrays
m$data_to_model(list('weight', 'height'))

# Define model ------------------------------------------------
model <- function(height, weight){
  # Parameters priors distributions
  s = bi.dist.uniform(0, 50, name = 's', shape =c(1))
  a = bi.dist.normal(178, 20,  name = 'a', shape = c(1))
  b = bi.dist.normal(0, 1, name = 'b', shape = c(1))
  
  # Likelihood
  bi.dist.normal(a + b * weight, s, obs = height)
}


# Run mcmc ------------------------------------------------
m$fit(model) # Optimize model parameters through MCMC sampling

# Summary ------------------------------------------------
m$summary()

```            

# BIR
Bayesian Inference for R 

Currently, the package provides:

+ Data manipulation:
    + One-hot encoding
    + Conversion of index variables
    + Scaling
      
+ Models (Using Numpyro):
  
    + [Linear Regression for continuous variable](Documentation/1.&#32;Linear&#32;Regression&#32;for&#32;continuous&#32;variable.qmd)
    + [Multiple continuous Variable](Documentation/2.&#32;Multiple&#32;continuous&#32;Variables.qmd)
    + [Interaction between variables](Documentation/3.&#32;Interaction&#32;between&#32;continuous&#32;variables.qmd)
    + [Categorical variable](Documentation/4.&#32;Categorical&#32;variable.qmd)
    + [Binomial model](Documentation/5.&#32;Binomial&#32;model.qmd)
    + [Beta binomial](Documentation/6.&#32;Beta&#32;binomial&#32;model.qmd)
    + [Poisson model](Documentation/7.&#32;Poisson&#32;model.qmd)
    + [Gamma-Poisson](Documentation/8.&#32;Gamma-Poisson.qmd)
    + [Multinomial](Documentation/9.&#32;Multinomial&#32;model.qmd)    
    + [Dirichlet model](Documentation/10.&#32;Dirichlet&#32;model&#32;(wip).qmd)
    + [Zero inflated](Documentation/11.&#32;Zero&#32;inflated.qmd)
    + [Varying intercept](Documentation/12.&#32;Varying&#32;intercepts.qmd)
    + [Varying slopes](Documentation/13.&#32Varying&#32slopes.qmd)
    + [Gaussian processes](Documentation/14.&#32;Gaussian&#32;processes&#32;(wip).qmd)  
    + [Measuring error](Documentation/15.&#32;Measuring&#32;error&#32;(wip).qmd) 
    + [Latent variable](Documentation/17.&#32;Latent&#32;variable&#32;(wip).qmd) 
    + [PCA](Documentation/18.&#32;PCA&#32;(wip).qmd) 
    + [Network model](Documentation/18.&#32;Network&#32;model.qmd) 
    + [Network with block model](Documentation/19.&#32;Network&#32;with&#32;block&#32;model.qmd)
    + [Network control for data collection biases ](Documentation/20.&#32;Network&#32;control&#32;for&#32;data&#32;collection&#32;biases&#32;(wip).qmd)

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
This package has been built following the Rethinking Classes of 2024. Each week, new approaches have been implemented and validated with the main example of the corresponding week. All models can be found in the following [Jupyter notebook](https://github.com/BGN-for-ASNA/BI/blob/main/Test/1.Rethinking_np.ipynb). 

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
m$data(paste(system.file(package = "BI"),"/data/Howell1.csv", sep = ''), sep=';')

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
  m$lk("y",  bi.dist.normal(a + b * weight, s), obs = height)
}


# Run mcmc ------------------------------------------------
m$run(model) # Optimize model parameters through MCMC sampling

# Summary ------------------------------------------------
m$summary()

```            

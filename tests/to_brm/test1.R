# %%
# Test the conversion from BI to brms
library(brms)
library(rstan)
library(abind)
library(reticulate)
source("../../R/to_brms.R")

cat("--- Testing BI to brms conversion ---\n")

# 1. Create mock BI data and "sampler"
# We'll simulate 2 chains, 100 samples each
set.seed(42)
n_chains <- 2
n_samples <- 100
n_warmup <- 50

# Mock samples dictionary (as if from reticulate/NumPyro)
mock_samples <- list(
  # BI often uses 'Intercept' or just 'alpha', 'beta'
  Intercept = array(rnorm(n_chains * n_samples, 1.5, 0.1), dim = c(n_chains, n_samples)),
  x = array(rnorm(n_chains * n_samples, 0.5, 0.05), dim = c(n_chains, n_samples)),
  sigma = array(abs(rnorm(n_chains * n_samples, 1, 0.1)), dim = c(n_chains, n_samples))
)

# Mock Log-probability samples
mock_extra <- list(
  potential_energy = array(runif(n_chains * n_samples, 10, 20), dim = c(n_chains, n_samples))
)

# A mock object that looks like a NumPyro MCMC object to R
mock_bi_fit <- list(
  get_samples = function(group_by_chain = TRUE) mock_samples,
  get_extra_fields = function(group_by_chain = TRUE) mock_extra,
  num_chains = n_chains,
  num_samples = n_samples,
  num_warmup = n_warmup,
  # Add these for automated extraction test
  formula = y ~ x,
  family = gaussian()
)

# 2. Define data (used for the skeleton)
df <- data.frame(
  x = rnorm(10),
  y = rnorm(10) # dummy y
)
mock_bi_fit$data <- df

# Mapping: 'x' from BI should be auto-mapped to 'b' in brms
# 'Intercept' from BI should be auto-mapped to 'b_Intercept' and 'Intercept' in brms
par_map <- list()

cat("Running conversion...\n")
fit_brms <- to_brms(
  bi_fit = mock_bi_fit,
  par_map = par_map,
  silent = TRUE,
  refresh = 0
)
cat("\n--- Brmsfit Variables ---\n")
print(variables(fit_brms))

cat("\n--- Summary of converted object ---\n")
summary_output <- summary(fit_brms)
print(summary_output)

# Check if fixed effects were extracted
if (!is.null(fixef(fit_brms))) {
    cat("\nFixed Effects extracted successfully!\n")
} else {
    cat("\nERROR: Fixed Effects NOT extracted.\n")
}

cat("\n--- Variance-Covariance Matrix ---\n")
print(vcov(fit_brms))

# Test plotting (to a file since no display)
cat("\n--- Testing plotting ---\n")
pdf("test_plots.pdf")
plot(fit_brms)
dev.off()
cat("Plots saved to test_plots.pdf\n")

cat("\nConversion and basic diagnostics successful!\n")

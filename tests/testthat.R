# Runs the testthat suite during R CMD check. Tests needing Python skip
# themselves when the 'BayesForge' Python package is not available.
library(testthat)
library(BayesForge)

test_check("BayesForge")

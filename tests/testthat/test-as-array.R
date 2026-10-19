# Tier 1.2 -- as.array(m) for bayesplot: iterations x chains x parameters.
if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)
  np <- reticulate::import("numpy", convert = FALSE)

  m <- importBF(platform = "cpu", rand_seed = 4, print_devices_found = FALSE)
  x <- seq(-1, 1, length.out = 30)
  m$fit(bf_model(function(x, y) {
          b <- bf.dist.normal(0, 2, shape = c(2L))
          s <- bf.dist.exponential(1)
          bf.dist.normal(b[0] + b[1] * x, s, obs = y)
        }),
        obs = list(x = jnp$asarray(x), y = jnp$asarray(0.5 + 2 * x)),
        num_warmup = 100L, num_samples = 60L, num_chains = 3L, progress_bar = FALSE)
  arr <- as.array(m)

  test_that("as.array(m) is iterations x chains x parameters with bf_draws names", {
    expect_identical(dim(arr), c(60L, 3L, 3L))
    expect_identical(dimnames(arr)$variable, c("b[1]", "b[2]", "s"))
    by_chain <- function(k) reticulate::py_to_r(np$asarray(m$posteriors_by_chain[[k]]))  # chains x draws (x 2)
    s <- by_chain("s")
    b <- by_chain("b")
    for (cc in 1:3) {
      expect_identical(arr[, cc, "s"], s[cc, ])
      expect_identical(arr[, cc, "b[2]"], b[cc, , 2])
    }
  })

  test_that("bayesplot accepts it", {
    skip_if_not_installed("bayesplot")
    expect_s3_class(bayesplot::mcmc_trace(arr), "ggplot")
    expect_s3_class(bayesplot::mcmc_intervals(arr), "ggplot")
  })
}

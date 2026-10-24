if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  test_that("bf_sample turns m$sample()'s JAX arrays into plain R arrays", {
    m <- importBF(platform = "cpu", rand_seed = 1, print_devices_found = FALSE)
    t <- seq(-1, 1, length.out = 25)
    y_obs <- 1 + 2 * t
    model <- function(t, y) {
      a <- bf.dist.normal(0, 5, name = "a")
      b <- bf.dist.normal(0, 5, name = "b")
      s <- bf.dist.exponential(1, name = "s")
      bf.dist.normal(a + b * t, s, obs = y)
    }
    m$fit(model, obs = list(t = jnp$asarray(t), y = jnp$asarray(y_obs)),
          num_warmup = 100L, num_samples = 100L, num_chains = 1L, progress_bar = FALSE)

    # an explicit seed gives identical draws on every call (see bf.sample docs)
    raw <- m$sample(seed = 42L)   # the untouched JAX result
    pp <- bf_sample(m, seed = 42L)

    expect_identical(names(pp), "y")
    expect_true(is.numeric(pp$y))
    shape <- reticulate::py_to_r(raw$y$shape)   # don't hard-code the draw count
    expect_identical(dim(pp$y), as.integer(unlist(shape)))
    expect_identical(dim(pp$y)[2], 25L)         # one column per observation

    expect_identical(as.array(raw$y), pp$y)
    expect_identical(as.matrix(raw$y), pp$y)
    expect_identical(.bf_jax_as_vector(raw$y), as.vector(pp$y))
  })

  test_that(".bf_jax_to_r leaves R values untouched", {
    expect_identical(.bf_jax_to_r(3), 3)
    expect_identical(.bf_jax_to_r("a"), "a")
  })
}

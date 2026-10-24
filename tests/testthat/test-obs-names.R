test_that(".bf_obs_name uses the variable passed as obs", {
  expect_identical(.bf_obs_name(quote(y)), "y")
  expect_identical(.bf_obs_name(quote(y - 1)), "obs")
  expect_identical(.bf_obs_name(quote(d$y)), "obs")
})

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  test_that("obs = y names the observed site y; a latent x no longer collides", {
    m <- importBF(platform = "cpu", rand_seed = 1, print_devices_found = FALSE)
    t <- seq(-1, 1, length.out = 25)
    y_obs <- 1 + 2 * t
    model <- function(t, y) {
      x <- bf.dist.normal(0, 5, name = "x")   # a latent called x
      s <- bf.dist.exponential(1, name = "s")
      bf.dist.normal(x + 0 * t, s, obs = y)   # used to be named "x" too
    }
    m$fit(model, obs = list(t = jnp$asarray(t), y = jnp$asarray(y_obs)),
          num_warmup = 100L, num_samples = 100L, num_chains = 1L, progress_bar = FALSE)
    expect_setequal(names(m$posteriors), c("x", "s"))
    sites <- reticulate::py_to_r(m$`_observed_sites`(m$data_on_model))
    expect_identical(names(sites), "y")
  })
}

test_that(".bf_infer translates enum = TRUE into NumPyro's infer dict", {
  none <- reticulate::py_none()
  expect_identical(.bf_infer(none, FALSE), none)
  expect_identical(.bf_infer(list(a = 1), FALSE), list(a = 1))
  expect_identical(.bf_infer(none, TRUE), list(enumerate = "parallel"))
  # an explicit infer wins key by key
  expect_identical(.bf_infer(list(enumerate = "sequential"), TRUE), list(enumerate = "sequential"))
  expect_identical(.bf_infer(list(obs_mask = TRUE), TRUE),
                   list(enumerate = "parallel", obs_mask = TRUE))
})

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  set.seed(0)
  z_true <- rbinom(30, 1, 0.4)
  y_mix <- ifelse(z_true == 0, rnorm(30, -2, 0.5), rnorm(30, 2, 0.5))

  # Two-component mixture; `mark` chooses how the assignment z is marked.
  fit_mixture <- function(mark, sampler = "nuts", ...) {
    m <- importBF(platform = "cpu", rand_seed = 123, print_devices_found = FALSE)
    model <- function(y) {
      w <- bf.dist.dirichlet(c(1, 1), name = "w")
      with(m$dist$plate("components", 2L), mu <- bf.dist.normal(0, 5, name = "mu"))
      sigma <- bf.dist.exponential(1, name = "sigma")
      with(m$dist$plate("N", 30L), {
        z <- switch(mark,
          enum  = bf.dist.categorical(probs = w, name = "z", enum = TRUE),
          infer = bf.dist.categorical(probs = w, name = "z", infer = list(enumerate = "parallel")),
          none  = bf.dist.categorical(probs = w, name = "z"))
        bf.dist.normal(jnp$take(mu, z), sigma, obs = y)
      })
    }
    args <- list(model, obs = list(y = jnp$asarray(y_mix)), num_warmup = 200L, num_samples = 200L,
                 num_chains = 1L, progress_bar = FALSE, ...)
    if (sampler != "nuts") args$sampler <- sampler  # older BayesForge has no sampler=
    do.call(m$fit, args)
    m
  }
  draws <- function(m, site) reticulate::py_to_r(np$asarray(m$posteriors[[site]]))
  np <- reticulate::import("numpy", convert = FALSE)
  fit_args <- reticulate::py_eval(
    "list(__import__('inspect').signature(__import__('BayesForge').bf.fit).parameters)")

  test_that("enum = TRUE gives exactly the infer-dict posterior and sums z out", {
    a <- fit_mixture("enum")
    b <- fit_mixture("infer")
    expect_false("z" %in% names(a$posteriors))
    for (site in c("w", "mu", "sigma")) expect_identical(draws(a, site), draws(b, site))
  })

  test_that("m$fit(sampler = ...) reaches the Python sampler", {
    skip_if_not("sampler" %in% fit_args, "installed BayesForge has no fit(sampler=)")
    m <- fit_mixture("none", sampler = "discrete_gibbs")
    expect_identical(reticulate::py_to_r(m$sampler_name), "discrete_gibbs")
    expect_true("z" %in% names(m$posteriors))  # sampled, not summed out
  })
}

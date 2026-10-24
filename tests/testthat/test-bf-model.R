test_that("bf_model() adds name = to assigned distribution calls", {
  f <- bf_model(function(x, y) {
    a <- bf.dist.normal(0, 1)
    b = bf.dist.normal(0, 1, name = "slope")       # explicit name kept
    with(m$dist$plate("g", 3L), u <- m$dist$normal(0, 1))
    s <- bf.dist.exponential(1)
    z <- x[, 1]                                     # not a distribution
    bf.dist.normal(a + b * x, s, obs = y)           # obs: named by the wrapper
  })
  b <- body(f)
  expect_identical(b[[2]][[3]]$name, "a")
  expect_identical(b[[3]][[3]]$name, "slope")
  expect_identical(b[[4]][[3]][[3]]$name, "u")
  expect_identical(b[[5]][[3]]$name, "s")
  expect_identical(deparse(b[[6]]), "z <- x[, 1]")
  expect_null(b[[7]]$name)
  expect_error(bf_model(1), "needs a function")
})

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  fit_lr <- function(model) {
    m <- importBF(platform = "cpu", rand_seed = 5, print_devices_found = FALSE)
    x <- seq(-1, 1, length.out = 30)
    m$fit(model, obs = list(x = jnp$asarray(x), y = jnp$asarray(0.5 + 2 * x)),
          num_warmup = 100L, num_samples = 100L, num_chains = 1L, progress_bar = FALSE)
    m
  }
  draws <- function(m, k) reticulate::py_to_r(reticulate::import("numpy")$asarray(m$posteriors[[k]]))

  test_that("bf_model() gives the same posterior as hand-written names", {
    auto <- fit_lr(bf_model(function(x, y) {
      a <- bf.dist.normal(0, 2)
      b <- bf.dist.normal(0, 2)
      s <- bf.dist.exponential(1)
      bf.dist.normal(a + b * x, s, obs = y)
    }))
    manual <- fit_lr(function(x, y) {
      a <- bf.dist.normal(0, 2, name = "a")
      b <- bf.dist.normal(0, 2, name = "b")
      s <- bf.dist.exponential(1, name = "s")
      bf.dist.normal(a + b * x, s, obs = y)
    })
    expect_setequal(names(auto$posteriors), c("a", "b", "s"))
    for (k in c("a", "b", "s")) expect_identical(draws(auto, k), draws(manual, k))
  })
}

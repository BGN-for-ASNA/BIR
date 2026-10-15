if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(testthat)
  library(BayesForge)

  fit_small <- function() {
    m <- importBF(platform = "cpu", rand_seed = 11, print_devices_found = FALSE)
    dat <- data.frame(x = seq(-1, 1, length.out = 30))
    dat$y <- 0.5 + 2 * dat$x + 0.1 * cos(7 * dat$x)
    invisible(m$data(dat))  # an R data.frame passed directly
    m$data_to_model(list("x", "y"))
    model <- function(x, y) {
      a <- bf.dist.normal(0, 2, name = "a", shape = c(2L))
      s <- bf.dist.exponential(1, name = "s")
      # Indexing inside model code is passed to Python: 0-based.
      bf.dist.normal(a[0] + a[1] * x, s, obs = y)
    }
    m$fit(model, num_samples = 50L, num_warmup = 50L, num_chains = 2L, progress_bar = FALSE)
    m
  }
  m <- fit_small()

  test_that("bf_draws returns one row per draw with posterior-style columns", {
    d <- bf_draws(m)
    expect_equal(nrow(d), 100)
    expect_equal(names(d), c(".chain", ".iteration", ".draw", "a[1]", "a[2]", "s"))
    expect_equal(d$.chain, rep(1:2, each = 50))
    expect_equal(d$.iteration, rep(1:50, times = 2))
  })

  test_that("draws land in the right chain/iteration/element", {
    np <- reticulate::import("numpy", convert = FALSE)
    a <- reticulate::py_to_r(np$asarray(m$sampler$get_samples(group_by_chain = TRUE)$a))
    d <- bf_draws(m)
    # chain 2, iteration 7, element 2
    expect_equal(d[d$.chain == 2 & d$.iteration == 7, "a[2]"], a[2, 7, 2])
  })

  test_that("as.data.frame() and data input from R work", {
    expect_identical(as.data.frame(m), bf_draws(m))
    expect_equal(mean(bf_draws(m)[["a[2]"]]), 2, tolerance = 0.1)
  })

  if (requireNamespace("posterior", quietly = TRUE)) {
    test_that("posterior::as_draws_df() works on a fitted model", {
      dd <- posterior::as_draws_df(m)
      expect_s3_class(dd, "draws_df")
      expect_equal(posterior::nchains(dd), 2)
      expect_true(all(c("a[1]", "a[2]", "s") %in% posterior::variables(dd)))
    })
  }
}

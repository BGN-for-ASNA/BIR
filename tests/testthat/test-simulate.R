# R side of BayesForge.Parallel.simulate (R/simulate.R): grid(), scheduling,
# seed derivation, error handling. Only test 10 starts JAX.

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(testthat)
  library(BayesForge)

  m <- importBF(print_devices_found = FALSE)
  sim <- reticulate::import("BayesForge.Parallel.simulate")

  test_that("grid expands parameters x reps into a data.frame", {
    g <- m$grid(N = c(100, 500), b_true = c(-0.8, 0, 0.8), reps = 4)
    expect_s3_class(g, "data.frame")
    expect_equal(nrow(g), 24L)
    expect_true(all(c("N", "b_true", "sim", "rep") %in% names(g)))
    # sim is 0-based, as in Python
    expect_equal(g$sim, 0:23)
  })

  test_that("process backend matches serial and runs off-process", {
    f <- function(N, seed) list(twice = 2 * N, s = seed)
    res_s <- m$run_simulations(f, m$grid(N = 1:4), backend = "serial")
    res_p <- m$run_simulations(f, m$grid(N = 1:4), workers = 2)
    rownames(res_s) <- NULL
    rownames(res_p) <- NULL
    expect_equal(res_p[c("sim", "N", "seed", "twice")],
                 res_s[c("sim", "N", "seed", "twice")])
    # two distinct worker pids, neither the test process
    expect_equal(length(unique(res_p$worker)), 2L)
    expect_false(any(res_p$worker == Sys.getpid()))
  })

  test_that("seed is derived from the run seed and the sim index", {
    f <- function(N, seed) list(s = seed)
    res_s <- m$run_simulations(f, m$grid(N = 1:4), backend = "serial")
    expect_equal(res_s$seed[res_s$sim == 3], sim$`_derive_seed`(0L, 3L))
  })

  test_that("scenario keys reach the function through ...", {
    res <- m$run_simulations(function(...) {
      list(has_sim = "sim" %in% names(list(...)))
    }, m$grid(N = 1:2), workers = 2)
    expect_true(all(res$has_sim))
  })

  test_that("failures are recorded per row, serial and process", {
    f <- function(N) {
      if (N == 2) stop("boom")
      list(ok = TRUE)
    }
    runs <- list(
      suppressMessages(m$run_simulations(f, m$grid(N = 1:3), backend = "serial")),
      suppressMessages(m$run_simulations(f, m$grid(N = 1:3), workers = 2))
    )
    for (res in runs) {
      bad <- res[!is.na(res$error), ]
      expect_equal(nrow(bad), 1L)
      expect_equal(bad$N, 2L)
      expect_match(bad$error, "boom")
    }
  })

  test_that("on_error = 'raise' stops on the first failure", {
    f <- function(N) {
      if (N == 2) stop("boom")
      list(ok = TRUE)
    }
    expect_error(m$run_simulations(f, m$grid(N = 1:3), backend = "serial",
                                   on_error = "raise"), "boom")
    expect_error(m$run_simulations(f, m$grid(N = 1:3), workers = 2,
                                   on_error = "raise"), "boom")
  })

  test_that("fn output is captured and reported per worker", {
    f <- function(N) {
      cat("hello", N, "\n")
      message("msg", N)
      list(v = N)
    }
    g <- m$grid(N = 1:2)
    out <- capture.output(res <- m$run_simulations(f, g, backend = "serial"))
    expect_true(any(grepl("worker main", out)))
    expect_true(any(grepl("hello 1", out)))
    expect_equal(res$v, 1:2)
    out <- capture.output(res <- m$run_simulations(f, g, workers = 2))
    expect_true(any(grepl("worker [12] ", out)))
    expect_true(any(grepl("hello 2", out)))
    expect_true(any(grepl("msg2", out)))
    expect_equal(res$v, 1:2)
  })

  test_that("progress = FALSE is silent", {
    expect_silent(m$run_simulations(function(N) list(v = N), m$grid(N = 1:2),
                                    backend = "serial", progress = FALSE))
  })

  test_that("output before an error is kept and the first error is reported", {
    f <- function(N) {
      cat("before\n")
      stop("boom")
    }
    expect_message(
      out <- capture.output(m$run_simulations(f, m$grid(N = 1), backend = "serial")),
      "first \\(sim 0\\): boom"
    )
    expect_true(any(grepl("before", out)))
    expect_true(any(grepl("ERROR: boom", out)))
  })

  test_that("an empty list result keeps its scenario row", {
    res <- m$run_simulations(function(N) list(), m$grid(N = 1:3), backend = "serial")
    expect_equal(nrow(res), 3L)
    expect_true(all(is.na(res$error)))
  })

  test_that("num_chains is injected into every scenario", {
    res <- m$run_simulations(function(num_chains, ...) list(nc = num_chains),
                             m$grid(N = 1:2), num_chains = 1, workers = 2)
    expect_true(all(res$nc == 1))
  })

  test_that("export ships a helper from the caller's frame", {
    h <- function(x) x + 1
    f <- function(N) list(v = h(N))
    res <- m$run_simulations(f, m$grid(N = 1:3), workers = 2, export = "h")
    expect_equal(res$v, res$N + 1)
  })

  test_that("returning the bf object is refused", {
    res <- suppressMessages(
      m$run_simulations(function(N) m, m$grid(N = 1), backend = "serial")
    )
    expect_match(res$error, "cannot cross a process boundary")
  })

  test_that("m keeps its Python methods", {
    expect_true(is.function(m$fit))
    expect_true(is.function(m$summary))
  })

  test_that("JAX fits run in parallel workers", {
    skip_on_cran()
    # Linear regression from Documentation/start/Parallel simulations.qmd (R tab).
    fit_one <- function(N, b_true, seed) {
      library(BayesForge)
      m <- importBF(rand_seed = seed, print_devices_found = FALSE)

      # --- simulate one dataset ------------------------------------
      x <- bf.dist.normal(0.0, 1.0, sample = TRUE, shape = N)
      y <- bf.dist.normal(1.5 + b_true * x, 1.0, sample = TRUE)

      # --- the model -------------------------------------------------
      linreg <- function(x, y) {
        a <- bf.dist.normal(0.0, 5.0, name = "a")
        b <- bf.dist.normal(0.0, 5.0, name = "b")
        s <- bf.dist.exponential(1.0, name = "s")
        bf.dist.normal(a + b * x, s, name = "y", obs = y)
      }

      # --- fit, then report the estimate ------------------------------
      m$fit(linreg, obs = list(x = x, y = y),
            num_warmup = 100L, num_samples = 100L, num_chains = 1L,
            progress_bar = FALSE)
      est <- m$summary()
      list(b_hat = est["b", "mean"])
    }
    res <- m$run_simulations(fit_one, m$grid(N = c(50, 100), b_true = 0.5),
                             workers = 2)
    expect_equal(nrow(res), 2L)
    expect_true(all(is.na(res$error)))
    expect_true(all(is.finite(res$b_hat)))
  })
}

# Tier 1.1 verification: R `loo` object (`loo::loo(m)`).
#
# Roadmap: S3 method `loo.BayesForge` in BFR: pointwise log-lik
# (iterations x chains x obs) from BF, then `loo::loo.array` with `r_eff`.
# Verification: elpd_loo from R vs `m$diag$loo()` in Python (|Δ| < 0.1 on a
# linear regression); `loo::loo_compare` runs on 2 models.

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge") &&
    requireNamespace("loo", quietly = TRUE)) {
  library(testthat)
  library(BayesForge)

  fit_lin <- function(intercept_only = FALSE, seed = 11L) {
    m <- importBF(platform = "cpu", rand_seed = seed, print_devices_found = FALSE)
    dat <- data.frame(x = seq(-1, 1, length.out = 30))
    dat$y <- 0.5 + 2 * dat$x + 0.1 * cos(7 * dat$x)
    # m$data() takes a CSV path on the released BF; newer BF also accepts a
    # data.frame directly. A temp CSV works on both.
    csv <- tempfile(fileext = ".csv")
    utils::write.csv(dat, csv, row.names = FALSE)
    invisible(m$data(csv))
    m$data_to_model(list("x", "y"))
    if (intercept_only) {
      model <- function(x, y) {
        a <- bf.dist.normal(0, 2, name = "a")
        s <- bf.dist.exponential(1, name = "s")
        bf.dist.normal(a, s, obs = y)
      }
    } else {
      model <- function(x, y) {
        a <- bf.dist.normal(0, 2, name = "a", shape = c(2L))
        s <- bf.dist.exponential(1, name = "s")
        # Indexing inside model code is passed to Python: 0-based.
        bf.dist.normal(a[0] + a[1] * x, s, obs = y)
      }
    }
    m$fit(model, num_samples = 100L, num_warmup = 100L, num_chains = 2L,
          progress_bar = FALSE)
    m
  }

  test_that("loo(m) matches Python elpd_loo within 0.1", {
    m <- fit_lin()
    l1 <- loo::loo(m)
    expect_s3_class(l1, "loo")
    expect_equal(dim(l1$pointwise)[1], 30)  # one row per observation
    py_elpd <- reticulate::py_to_r(m$diag$loo()$to_dict())[["elpd_loo"]]
    r_elpd <- l1$estimates["elpd_loo", "Estimate"]
    expect_lt(abs(r_elpd - py_elpd), 0.1)
  })

  test_that("loo_compare runs on two models", {
    m1 <- fit_lin(intercept_only = FALSE)
    m2 <- fit_lin(intercept_only = TRUE)
    cmp <- loo::loo_compare(loo::loo(m1), loo::loo(m2))
    expect_true("elpd_diff" %in% colnames(cmp))
    # The slope model predicts better out of sample.
    expect_gt(cmp["model2", "elpd_diff"], -Inf)
    expect_lt(cmp["model2", "elpd_diff"], 0)
  })
}

# Live bayesplot cross-check of the PPC plots exposed as `m$diag$ppc_*`.
#
# Roadmap: the Python-side figures of `BayesForge/Diagnostic/ppc.py` are read
# back in R through reticulate, and the numbers they carry are compared against
# R **bayesplot 1.16.0** (`ppc_intervals_data`, `ppc_ribbon_data`,
# `ppc_bars_data`) and **rstantools 2.7.1** `loo_pit` (with `loo::psis`
# weights) -- the same mapping as BF `Test/2.Diagnostic_plots/README.md` and
# `make_reference.R`, but computed live from one fitted model instead of a
# fixture. This complements the detailed fixture tests of the Python repo
# (`Test/2.Diagnostic_plots/test_ppc_vs_bayesplot.py`).
#
# What is verified: the interval bounds of `ppc_intervals` / `ppc_ribbon` /
# `ppc_intervals_sorted` (type-7 quantiles at (1-prob_outer)/2, (1-prob)/2,
# 0.5, (1+prob)/2, (1+prob_outer)/2), the test statistics of `ppc_stat` and
# `ppc_stat_2d`, `colMeans(yrep)` in `ppc_error_scatter`, the mean residuals in
# `ppc_error_hist`, observed/expected counts and the 5%/95% count quantiles in
# `ppc_bars` and `ppc_rootogram`, and the LOO-PIT values behind
# `ppc_loo_pit(log_likelihood = ...)`.
#
# `np.percentile` (linear interpolation) equals `stats::quantile(type = 7)`,
# which is what bayesplot uses, so every quantity that is exact arithmetic is
# compared at `tolerance = 1e-7`. The same `y` and `yrep` are passed to both
# sides, so a failure is an algorithmic difference, never rounding.
#
# Known difference (documented, skipped below, same reason as the Python
# test): `ppc_error_hist` plots 30 mean residuals instead of bayesplot's
# 200 x 30 per-draw errors. The LOO-PIT PSIS difference that the Python test
# xfails (ArviZ-style PSIS vs `loo::psis`, max |difference| 1.15e-8 on the
# fixture) does NOT show up on this fit -- PSIS smoothing is inactive here, so
# the two agree to 1.1e-15 and the comparison is asserted rather than skipped.
#
# The figures are never printed, so no browser is opened during the tests.
#
# Not compared here (they need bandwidth / binning / sampled-draw conventions
# that differ by design, and the Python fixture tests cover them): ppc_density,
# ppc_hist, ppc_boxplot, ppc_scatter. The two count plots use one extra small
# Poisson regression, because ppc_bars and ppc_rootogram need a discrete
# outcome.

if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge") &&
    requireNamespace("loo", quietly = TRUE) &&
    requireNamespace("bayesplot", quietly = TRUE) &&
    requireNamespace("rstantools", quietly = TRUE)) {
  library(testthat)
  library(BayesForge)

  np <- reticulate::import("numpy", convert = FALSE)

  ## ---- helpers: reading a plotly figure from R ------------------------------
  # fig$data is a Python tuple of trace objects; a trace field is a numpy
  # array, a Python tuple (plotly's segment lists hold None separators, which
  # become NaN) or None when unset.
  py_get <- function(obj, field) {
    v <- tryCatch(reticulate::py_get_attr(obj, field), error = function(e) NULL)
    if (is.null(v) || inherits(v, "python.builtin.NoneType") ||
        reticulate::py_is_null_xptr(v)) {
      return(NULL)
    }
    v
  }
  traces <- function(fig) reticulate::py_to_r(fig$data)
  trace_names <- function(tr) {
    vapply(tr, function(t) {
      nm <- py_get(t, "name")
      if (is.null(nm)) "" else as.character(nm)
    }, character(1))
  }
  # k selects the k-th trace with that name (ppc_intervals_sorted draws both
  # intervals with prob = prob_outer, so both traces are named "90% interval").
  trace_by_name <- function(tr, name, k = 1L) {
    hit <- which(trace_names(tr) == name)
    if (length(hit) < k) {
      stop("no trace #", k, " named '", name, "'; traces: ",
           paste(trace_names(tr), collapse = " | "))
    }
    tr[[hit[[k]]]]
  }
  trace_field <- function(tr, field) {
    v <- py_get(tr, field)
    if (is.null(v)) return(NULL)
    as.numeric(reticulate::py_to_r(np$asarray(v, dtype = "float64")))
  }

  # (ll, l, m, h, hh, observed, x) per observation from a ppc_intervals /
  # ppc_ribbon / ppc_intervals_sorted figure. The two interval traces are
  # vertical segments (lo, hi, None, ...) in ppc_intervals and closed polygons
  # y = c(hi, rev(lo)) in ppc_ribbon; rows are sorted by x.
  lo_hi <- function(trace, n) {
    v <- trace_field(trace, "y")
    fill <- py_get(trace, "fill")
    fill <- if (is.null(fill)) "" else as.character(fill)
    if (identical(fill, "toself")) {
      return(list(lo = rev(v[seq.int(n + 1L, 2L * n)]), hi = v[seq_len(n)]))
    }
    list(lo = v[seq.int(1L, 3L * n, by = 3L)], hi = v[seq.int(2L, 3L * n, by = 3L)])
  }
  interval_bounds <- function(fig, n, outer_name, inner_name,
                              k_outer = 1L, k_inner = 1L) {
    tr <- traces(fig)
    outer_i <- lo_hi(trace_by_name(tr, outer_name, k_outer), n)
    inner_i <- lo_hi(trace_by_name(tr, inner_name, k_inner), n)
    med <- trace_by_name(tr, "Median prediction of y")
    obs <- trace_by_name(tr, "y (observed)")
    list(x = trace_field(med, "x"), ll = outer_i$lo, l = inner_i$lo,
         m = trace_field(med, "y"), h = inner_i$hi, hh = outer_i$hi,
         obs = trace_field(obs, "y"))
  }
  # layout shapes (ppc_stat draws T(y) as a vertical line): plain JSON values.
  layout_json <- function(fig) reticulate::py_to_r(fig$layout$to_plotly_json())

  # Scott-bandwidth Gaussian KDE of values in [0, 1], reflected at 0 and 1
  # (bayesplot's boundary correction) -- the density ppc_loo_pit plots.
  reflected_kde <- function(v, xs) {
    n <- length(v)
    bw <- stats::sd(v) * n^(-0.2)
    pts <- c(v, -v, 2 - v)
    z <- outer(xs, pts, `-`) / bw
    rowSums(exp(-0.5 * z * z)) / (n * bw * sqrt(2 * pi))
  }

  ## ---- one fit per outcome, reused by every test ----------------------------
  .shared <- new.env(parent = emptyenv())

  fit_with <- function(model, y, seed) {
    m <- importBF(platform = "cpu", rand_seed = seed, print_devices_found = FALSE)
    invisible(m$data(data.frame(x = seq(-1, 1, length.out = 30), y = y)))
    m$data_to_model(list("x", "y"))
    m$fit(model, num_samples = 200L, num_warmup = 200L, num_chains = 1L,
          progress_bar = FALSE)
    m
  }

  # the same fit_lin() as test-loo.R: 30 observations, 1 chain
  shared_lin <- function() {
    if (is.null(.shared$lin)) {
      model <- function(x, y) {
        a <- bf.dist.normal(0, 2, name = "a", shape = c(2L))
        s <- bf.dist.exponential(1, name = "s")
        bf.dist.normal(a[0] + a[1] * x, s, obs = y)
      }
      x <- seq(-1, 1, length.out = 30)
      y <- 0.5 + 2 * x + 0.1 * cos(7 * x)
      m <- fit_with(model, y, seed = 7L)
      # yrep as an R matrix (draws x obs); THE SAME y and yrep go to both sides
      yrep <- reticulate::py_to_r(np$asarray(m$diag$get_yrep(seed = 123L)))
      stopifnot(identical(dim(yrep), c(200L, 30L)))
      .shared$lin <- list(m = m, x = x, y = y, yrep = yrep)
    }
    .shared$lin
  }

  # counts for ppc_bars / ppc_rootogram: a small Poisson regression
  shared_pois <- function() {
    if (is.null(.shared$pois)) {
      model <- function(x, y) {
        a <- bf.dist.normal(0, 2, name = "a", shape = c(2L))
        bf.dist.poisson(rate = exp(a[0] + a[1] * x), obs = y)
      }
      x <- seq(-1, 1, length.out = 30)
      y <- as.integer(round(1 + 3 * exp(1.2 * x)))
      m <- fit_with(model, y, seed = 7L)
      yrep <- reticulate::py_to_r(np$asarray(m$diag$get_yrep(seed = 5L)))
      stopifnot(identical(dim(yrep), c(200L, 30L)))
      .shared$pois <- list(m = m, y = y, yrep = yrep)
    }
    .shared$pois
  }

  # pointwise log-likelihood as a (draws x obs) matrix in the draw order of
  # get_yrep() (iterations vary fastest within a chain, 1 chain here)
  log_lik_matrix <- function(m) {
    ll <- BayesForge:::.bf_log_lik_array(m)   # iterations x chains x obs
    matrix(ll, nrow = dim(ll)[1] * dim(ll)[2])
  }

  # the LOO-PIT vector ppc_loo_pit computes: its own ArviZ-style PSIS weights
  # (jax_diagnostics._psis_weights, reff = 1), normalised per observation, and
  # the weighted CDF sum(w * 1{yrep <= y}).
  bf_loo_pit <- function(y, yrep, ll) {
    psis_w <- reticulate::import("BayesForge.Diagnostic.jax_diagnostics")$`_psis_weights`
    vapply(seq_along(y), function(i) {
      lw <- as.numeric(reticulate::py_to_r(
        np$asarray(psis_w(ll[, i], reff = 1.0)[[1]], dtype = "float64")))
      w <- exp(lw)
      sum(w / sum(w) * (yrep[, i] <= y[i]))
    }, numeric(1))
  }

  ## ---- 1. ppc_intervals -----------------------------------------------------
  test_that("ppc_intervals bounds match bayesplot ppc_intervals_data", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_intervals(y = s$y, yrep = s$yrep, x = s$x,
                                  prob = 0.5, prob_outer = 0.9)
    iv <- as.data.frame(bayesplot:::ppc_intervals_data(
      s$y, s$yrep, x = s$x, prob = 0.5, prob_outer = 0.9))
    # bayesplot returns rows in observation order, BayesForge sorts them by x
    xo <- order(s$x)
    b <- interval_bounds(fig, length(s$y), "90% interval", "50% interval")
    expect_equal(b$x, s$x[xo])
    expect_equal(b$ll, iv$ll[xo], tolerance = 1e-7)
    expect_equal(b$l, iv$l[xo], tolerance = 1e-7)
    expect_equal(b$m, iv$m[xo], tolerance = 1e-7)
    expect_equal(b$h, iv$h[xo], tolerance = 1e-7)
    expect_equal(b$hh, iv$hh[xo], tolerance = 1e-7)
    expect_equal(b$obs, s$y[xo], tolerance = 1e-7)
  })

  ## ---- 2. ppc_ribbon --------------------------------------------------------
  test_that("ppc_ribbon bounds match bayesplot ppc_ribbon_data", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_ribbon(y = s$y, yrep = s$yrep, x = s$x,
                               prob = 0.5, prob_outer = 0.9)
    rb <- as.data.frame(bayesplot:::ppc_ribbon_data(
      s$y, s$yrep, x = s$x, prob = 0.5, prob_outer = 0.9))
    xo <- order(s$x)
    # same .ppd_intervals_data bounds as ppc_intervals, drawn as polygons
    b <- interval_bounds(fig, length(s$y), "90% interval", "50% interval")
    expect_equal(b$x, s$x[xo])
    expect_equal(b$ll, rb$ll[xo], tolerance = 1e-7)
    expect_equal(b$l, rb$l[xo], tolerance = 1e-7)
    expect_equal(b$m, rb$m[xo], tolerance = 1e-7)
    expect_equal(b$h, rb$h[xo], tolerance = 1e-7)
    expect_equal(b$hh, rb$hh[xo], tolerance = 1e-7)
    expect_equal(b$obs, s$y[xo], tolerance = 1e-7)
  })

  ## ---- 3. ppc_intervals_sorted ---------------------------------------------
  test_that("ppc_intervals_sorted bounds match bayesplot on rows sorted by y", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_intervals_sorted(y = s$y, yrep = s$yrep, prob = 0.9)
    ord <- order(s$y)   # the reference ordering: observations sorted by y
    iv <- as.data.frame(bayesplot:::ppc_intervals_data(
      s$y[ord], s$yrep[, ord, drop = FALSE], x = s$x[ord],
      prob = 0.9, prob_outer = 0.9))
    # prob = prob_outer = 0.9, so both interval traces are named "90% interval"
    b <- interval_bounds(fig, length(s$y), "90% interval", "90% interval",
                         k_inner = 2L)
    expect_equal(b$obs, sort(s$y), tolerance = 1e-7)
    expect_equal(b$ll, iv$ll, tolerance = 1e-7)
    expect_equal(b$l, iv$l, tolerance = 1e-7)
    expect_equal(b$m, iv$m, tolerance = 1e-7)
    expect_equal(b$h, iv$h, tolerance = 1e-7)
    expect_equal(b$hh, iv$hh, tolerance = 1e-7)
  })

  ## ---- 4. ppc_stat (mean) ---------------------------------------------------
  test_that("ppc_stat(mean) matches bayesplot ppc_stat_data statistics", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_stat(y = s$y, yrep = s$yrep, stat = "mean")
    # reference: bayesplot's own ppc_stat_data(stat = "mean") (the same call
    # make_reference.R's stat_mean_y / stat_mean_yrep stand for), row "y" is
    # T(y), the yrep_1..yrep_200 rows are T(yrep) per draw
    st <- bayesplot::ppc_stat_data(s$y, s$yrep, stat = "mean")
    is_y <- as.character(st$variable) == "y"
    # T(y) is the observed vline, T(yrep) only through the KDE grid, whose
    # endpoints are min/max of the 200 per-draw statistics.
    expect_equal(layout_json(fig)$shapes[[1]]$x0, st$value[is_y],
                 tolerance = 1e-7)
    grid <- trace_field(trace_by_name(traces(fig),
                                      "mean of each prediction of y"), "x")
    t_rep <- st$value[!is_y]
    expect_equal(grid[[1L]], min(t_rep), tolerance = 1e-7)
    expect_equal(grid[[length(grid)]], max(t_rep), tolerance = 1e-7)
  })

  ## ---- 5. ppc_stat_2d -------------------------------------------------------
  test_that("ppc_stat_2d(mean, sd) matches bayesplot ppc_stat_2d statistics", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_stat_2d(y = s$y, yrep = s$yrep, stat1 = "mean",
                                stat2 = "sd")
    # reference: bayesplot's own ppc_stat_data(stat = c("mean", "sd")), the
    # same call bayesplot::ppc_stat_2d makes internally (value = mean,
    # value2 = sd; make_reference.R's stat_mean_* / stat_sd_* quantities)
    st <- bayesplot::ppc_stat_data(s$y, s$yrep, stat = c("mean", "sd"))
    is_y <- as.character(st$variable) == "y"
    tr <- traces(fig)
    rep <- trace_by_name(tr, "Predictions of y")
    obs <- trace_by_name(tr, "y (observed)")
    expect_equal(trace_field(rep, "x"), st$value[!is_y], tolerance = 1e-7)
    expect_equal(trace_field(rep, "y"), st$value2[!is_y], tolerance = 1e-7)
    expect_equal(trace_field(obs, "x"), st$value[is_y], tolerance = 1e-7)
    expect_equal(trace_field(obs, "y"), st$value2[is_y], tolerance = 1e-7)
  })

  ## ---- 6. ppc_error_scatter -------------------------------------------------
  test_that("ppc_error_scatter matches bayesplot ppc_error_scatter_avg", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_error_scatter(y = s$y, yrep = s$yrep)
    # reference: the data frame of bayesplot's own plot, as make_reference.R's
    # error_scatter_x / error_scatter_y quantities: p$data$value carries the
    # mean errors y - E[yrep] and p$data$y_obs the observed y, so BF's
    # x = E[yrep] = y_obs - value and y = y_obs (the fields the Python
    # fixture test test_ppc_error_scatter compares)
    p <- bayesplot::ppc_error_scatter_avg(s$y, s$yrep)
    pts <- trace_by_name(traces(fig), "observations")
    expect_equal(trace_field(pts, "x"), p$data$y_obs - p$data$value,
                 tolerance = 1e-7)
    expect_equal(trace_field(pts, "y"), p$data$y_obs, tolerance = 1e-7)
  })

  ## ---- 7. ppc_error_hist ----------------------------------------------------
  test_that("ppc_error_hist plots the mean residuals y - E[yrep]", {
    s <- shared_lin()
    fig <- s$m$diag$ppc_error_hist(y = s$y, yrep = s$yrep)
    resid_mean <- trace_field(traces(fig)[[1L]], "x")
    expect_equal(resid_mean, s$y - colMeans(s$yrep), tolerance = 1e-7)

    # known difference: bayesplot ppc_error_hist plots the per-draw errors
    # y - yrep[s, i] (rstantools::predictive_error, 200 x 30, faceted per
    # draw); BayesForge exposes only the 30 mean residuals, so the figure
    # cannot equal those 200 x 30 errors.
    skip(paste("known difference: ppc_error_hist plots one histogram of the 30",
               "mean residuals y - E[yrep], not bayesplot's 200 x 30 per-draw",
               "errors y - yrep[s, i]"))
  })

  ## ---- 8. ppc_bars ----------------------------------------------------------
  test_that("ppc_bars counts and 90% interval match bayesplot ppc_bars_data", {
    s <- shared_pois()
    fig <- s$m$diag$ppc_bars(y = s$y, yrep = s$yrep)
    bd <- as.data.frame(bayesplot:::ppc_bars_data(s$y, s$yrep, prob = 0.9,
                                                  freq = TRUE))
    tr <- traces(fig)
    obs <- trace_by_name(tr, "y (observed)")
    rep <- trace_by_name(tr, "Predictions of y: median count \u00b1 90% interval")
    expect_equal(trace_field(obs, "x"), as.numeric(bd$x))
    expect_equal(trace_field(obs, "y"), as.numeric(bd$y_obs), tolerance = 1e-7)

    pt <- trace_field(rep, "y")
    err <- py_get(rep, "error_y")
    lo <- pt - trace_field(err, "arrayminus")
    hi <- pt + trace_field(err, "array")
    # the 5%/95% count quantiles are bayesplot's l/h (type-7 quantiles)
    expect_equal(lo, bd$l, tolerance = 1e-7)
    expect_equal(hi, bd$h, tolerance = 1e-7)
    # ... and the point is bayesplot's median count column (bd$m)
    expect_equal(pt, bd$m, tolerance = 1e-7)
  })

  ## ---- 9. ppc_rootogram -----------------------------------------------------
  test_that("ppc_rootogram matches bayesplot ppc_rootogram (hanging, sqrt)", {
    s <- shared_pois()
    fig <- s$m$diag$ppc_rootogram(y = s$y, yrep = s$yrep)
    tr <- traces(fig)
    bar <- trace_by_name(tr, "\u221aObserved (hanging from \u221aExpected)")
    curve <- trace_by_name(tr, "\u221aExpected count (predictions of y)")
    # reference as in make_reference.R: xpos = 0:max(y, yrep), expected =
    # colMeans of the per-draw counts (bayesplot ppc_rootogram, style="hanging")
    cx <- 0:max(c(s$y, as.vector(s$yrep)))
    obs_ct <- tabulate(match(s$y, cx), nbins = length(cx))
    counts <- t(apply(s$yrep, 1L, function(r) tabulate(match(r, cx),
                                                       nbins = length(cx))))
    expected <- colMeans(counts)
    expect_equal(trace_field(bar, "x"), as.numeric(cx))
    expect_equal(trace_field(bar, "y"), sqrt(obs_ct), tolerance = 1e-7)
    # hanging bars: height sqrt(observed), based at sqrt(expected) - sqrt(observed)
    expect_equal(trace_field(bar, "base"), sqrt(expected) - sqrt(obs_ct),
                 tolerance = 1e-7)
    expect_equal(trace_field(curve, "y"), sqrt(expected), tolerance = 1e-7)
  })

  ## ---- 10. ppc_loo_pit ------------------------------------------------------
  test_that("ppc_loo_pit(log_likelihood) PIT matches rstantools::loo_pit", {
    s <- shared_lin()
    ll <- log_lik_matrix(s$m)
    expect_equal(dim(ll), dim(s$yrep))   # same draws as yrep
    fig <- s$m$diag$ppc_loo_pit(y = s$y, yrep = s$yrep, log_likelihood = ll)

    # the figure carries only the PIT density (last trace, on a fixed grid):
    # recompute the PIT vector through the PSIS path ppc_loo_pit uses and tie
    # it to the figure by checking the plotted density.
    pit <- bf_loo_pit(s$y, s$yrep, ll)
    tr <- traces(fig)
    curve <- trace_by_name(tr, "LOO-PIT of y")
    grid <- trace_field(curve, "x")
    expect_equal(grid, seq(0, 1, length.out = 300L))
    expect_equal(trace_field(curve, "y"), reflected_kde(pit, grid),
                 tolerance = 1e-7)

    # R reference: rstantools::loo_pit with loo::psis(-log_lik, r_eff = 1)
    # weights, as make_reference.R does. The Python test xfails this comparison
    # on its fixture (ArviZ-style PSIS vs loo::psis differ by 1.15e-8 there);
    # with this fit PSIS smoothing is inactive and both agree to ~1e-15, so it
    # is asserted here instead of skipped.
    psis <- loo::psis(-ll, r_eff = 1)
    set.seed(1)   # rstantools breaks ties with runif()
    pit_r <- as.numeric(rstantools::loo_pit(object = s$yrep, y = s$y,
                                            lw = stats::weights(psis)))
    expect_equal(pit, pit_r, tolerance = 1e-7)
  })
}

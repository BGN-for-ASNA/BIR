#' PSIS-LOO for a fitted BayesForge model
#'
#' S3 method for [loo::loo()] on a fitted BayesForge object (from
#' [importBF()] followed by `m$fit()`). The pointwise log-likelihood
#' (iterations \eqn{\times} chains \eqn{\times} observations) is taken from
#' BayesForge through ArviZ, then passed to [loo::loo.array()] with
#' MCMC-based relative effective sample sizes, exactly as
#' `loo::loo_compare()` expects.
#'
#' R and Python agree up to PSIS smoothing details: on a linear regression,
#' `elpd_loo` here matches `m$diag$loo()` in Python within 0.1.
#'
#' @param x A fitted BayesForge object.
#' @param ... Passed on to [loo::loo.array()] (e.g. `cores`, `save_psis`).
#'
#' @return A `"loo"` object (see [loo::loo.array()]), usable with
#'   [loo::loo_compare()].
#' @examples
#' \dontrun{
#' library(BayesForge)
#' m <- importBF()
#' # ... m$fit(model) ...
#' loo::loo(m)
#' loo::loo_compare(loo::loo(m1), loo::loo(m2))
#' }
#' @export
loo.BayesForge.Main.main.bf <- function(x, ...) {  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("loo::loo() on a BayesForge model requires the 'loo' package.", call. = FALSE)
  }
  ll <- .bf_log_lik_array(x)  # iterations x chains x observations
  S <- dim(ll)[1]
  C <- dim(ll)[2]
  N <- dim(ll)[3]
  if (C > 1L) {
    # R arrays are column-major: rows run over iterations first, then chains.
    ll_mat <- matrix(ll, nrow = S * C)
    r_eff <- loo::relative_eff(
      exp(ll_mat), chain_id = rep(seq_len(C), each = S)
    )
  } else {
    r_eff <- NULL
  }
  loo::loo(ll, r_eff = r_eff, cores = 1L, ...)
}

# `loo` is only Suggested, so the S3 method cannot be declared in NAMESPACE
# (loading would fail when `loo` is absent). Register it when the namespace
# loads and `loo` is available -- the same pattern used for
# `posterior::as_draws_df`.
if (requireNamespace("loo", quietly = TRUE)) {
  registerS3method(
    "loo", "BayesForge.Main.main.bf",
    loo.BayesForge.Main.main.bf,
    envir = asNamespace("loo")
  )
}

#' Pointwise log-likelihood of a fitted model as an R array
#'
#' Reads the `log_likelihood` group built by ArviZ from the fitted sampler
#' (`az$from_numpyro(m$sampler, log_likelihood = TRUE)`, the same quantity
#' Python's `m$diag$loo()` uses) and returns it as an R array with dimensions
#' iterations \eqn{\times} chains \eqn{\times} observations -- the layout
#' [loo::loo.array()] expects. Observed variables are concatenated along the
#' observation axis; trailing event dimensions are flattened per variable.
#'
#' @param m A fitted BayesForge object.
#' @return A numeric array, dimensions `c(n_iterations, n_chains, n_obs)`.
#' @keywords internal
.bf_log_lik_array <- function(m) {
  if (is.null(m$sampler) || inherits(m$sampler, "python.builtin.NoneType")) {
    stop("No posterior draws: fit the model first with m$fit().", call. = FALSE)
  }
  az <- reticulate::import("arviz")
  idata <- az$from_numpyro(m$sampler, log_likelihood = TRUE)
  ll_group <- idata$log_likelihood
  vnames <- reticulate::py_to_r(reticulate::import_builtins()$list(ll_group$data_vars))
  if (length(vnames) == 0L) {
    stop("ArviZ returned no log_likelihood variables for this model.", call. = FALSE)
  }
  slabs <- lapply(vnames, function(v) {
    arr <- reticulate::py_to_r(ll_group[[v]]$values)  # chain x draw x event...
    d <- dim(arr)
    if (is.null(d)) d <- c(1L, length(arr))
    C <- d[1]
    S <- d[2]
    K <- if (length(d) > 2L) prod(d[-(1:2)]) else 1L
    # (iterations, chains, obs): iterations vary fastest within a chain.
    out <- array(NA_real_, dim = c(S, C, K))
    flat <- array(arr, dim = c(C, S, K))
    for (k in seq_len(K)) out[, , k] <- t(flat[, , k])
    out
  })
  S <- dim(slabs[[1L]])[1]
  C <- dim(slabs[[1L]])[2]
  N <- sum(vapply(slabs, function(a) dim(a)[3], integer(1)))
  out <- array(unlist(slabs), dim = c(S, C, N))
  dimnames(out) <- list(
    iteration = seq_len(S),
    chain = seq_len(C),
    observation = seq_len(N)
  )
  out
}

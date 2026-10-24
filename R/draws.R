#' Posterior draws as an R data frame
#'
#' Extracts the posterior draws of a fitted BayesForge model into a plain R
#' `data.frame`, so they can be summarised, plotted or combined with any other
#' R tool (dplyr, ggplot2, bayesplot, loo, ...).
#'
#' Each row is one draw. The columns `.chain`, `.iteration` and `.draw` identify
#' the draw (the layout used by the **posterior** package), followed by one column
#' per scalar parameter element: a vector parameter `b` gives `b[1]`, `b[2]`, ...,
#' a matrix parameter `S` gives `S[1,1]`, `S[2,1]`, ...
#'
#' Note that these names follow the R (1-based) convention used by **posterior**,
#' **brms** and **bayesplot**, while indexing *inside model code* is passed to
#' Python and is 0-based: `a[0]` in the model is column `a[1]` here.
#'
#' @param m A fitted BayesForge object (from [importBF()] followed by `m$fit()`).
#' @param format `"data.frame"` (default, one row per draw), `"draws_df"`, which
#'   returns a `posterior::draws_df` (requires the **posterior** package), or
#'   `"long"`: one row per draw and parameter element, with columns `.chain`,
#'   `.iteration`, `.draw`, `.variable` (parameter name), `.index` (element,
#'   e.g. `"2"` or `"1,3"`; `NA` for a scalar) and `.value` -- the tidy layout
#'   used by ggplot2 / dplyr / tidybayes.
#' @param labels For `format = "long"`: a named list replacing the index of a
#'   vector parameter by labels, e.g. `list(alpha = levels(df$group))` when
#'   `alpha` is indexed by the 0-based codes of factor `group` (from `m$index()`).
#'
#' @return A `data.frame` (or a `posterior::draws_df`).
#' @examples
#' \dontrun{
#' m <- importBF()
#' # ... m$fit(model) ...
#' draws <- bf_draws(m)
#' colMeans(draws[, -(1:3)])
#' posterior::summarise_draws(bf_draws(m, "draws_df"))
#' bayesplot::mcmc_trace(as.array(m))   # iterations x chains x parameters
#' long <- bf_draws(m, "long", labels = list(alpha = levels(df$group)))
#' }
#' @export
bf_draws <- function(m, format = c("data.frame", "draws_df", "long"), labels = NULL) {
  format <- match.arg(format)
  if (is.null(m$sampler) || inherits(m$sampler, "python.builtin.NoneType")) {
    stop("No posterior draws: fit the model first with m$fit().", call. = FALSE)
  }
  np <- reticulate::import("numpy", convert = FALSE)
  # A named R list of (unconverted) JAX arrays, each (chains, draws, ...).
  samples <- m$sampler$get_samples(group_by_chain = TRUE)
  names_py <- names(samples)

  columns <- list()
  n_chains <- n_iter <- NULL
  for (name in names_py) {
    arr <- reticulate::py_to_r(np$asarray(samples[[name]]))
    d <- dim(arr)
    if (is.null(d)) d <- c(1L, length(arr))
    n_chains <- d[1]
    n_iter <- d[2]
    # Rows ordered by chain, then iteration (chain is the slowest index).
    rows <- function(slice) as.vector(t(matrix(slice, nrow = n_chains)))
    if (length(d) == 2) {
      columns[[name]] <- rows(arr)
    } else {
      trailing <- d[-(1:2)]
      K <- prod(trailing)
      # Flatten trailing dimensions column-major (first index fastest).
      flat <- array(arr, dim = c(n_chains, n_iter, K))
      idx <- as.matrix(expand.grid(lapply(trailing, seq_len)))
      for (k in seq_len(K)) {
        col <- paste0(name, "[", paste(idx[k, ], collapse = ","), "]")
        columns[[col]] <- rows(flat[, , k])
      }
    }
  }

  out <- data.frame(
    .chain = rep(seq_len(n_chains), each = n_iter),
    .iteration = rep(seq_len(n_iter), times = n_chains),
    .draw = seq_len(n_chains * n_iter),
    check.names = FALSE
  )
  out <- cbind(out, as.data.frame(columns, check.names = FALSE))

  if (format == "long") {
    return(.bf_draws_long(out, labels))
  }
  if (format == "draws_df") {
    if (!requireNamespace("posterior", quietly = TRUE)) {
      stop("format = \"draws_df\" requires the 'posterior' package.", call. = FALSE)
    }
    return(posterior::as_draws_df(out))
  }
  out
}

#' @rdname bf_draws
#' @param x A fitted BayesForge object.
#' @param row.names,optional,... Ignored; for compatibility with [as.data.frame()].
#' @export
as.data.frame.BayesForge.Main.main.bf <- function(x, row.names = NULL, optional = FALSE, ...) {
  bf_draws(x)
}

# posterior::as_draws_df() method, registered in .onLoad() when the
# 'posterior' package is installed (it is only suggested).
.as_draws_df_bf <- function(x, ...) {
  bf_draws(x, format = "draws_df")
}

# Wide draws (one column per parameter element) -> one row per draw and
# parameter element: .chain, .iteration, .draw, .variable, .index, .value.
# `labels`: named list, e.g. list(alpha = levels(df$group)), replacing the
# 1-based index of a vector parameter by these labels.
.bf_draws_long <- function(wide, labels = NULL) {
  ids <- c(".chain", ".iteration", ".draw")
  cols <- setdiff(names(wide), ids)
  variable <- sub("\\[.*$", "", cols)
  index <- ifelse(grepl("\\[", cols), sub("^[^[]*\\[(.*)\\]$", "\\1", cols), NA_character_)
  for (v in names(labels)) {
    sel <- variable == v
    if (!any(sel)) stop("labels: no parameter called ", v, call. = FALSE)
    pos <- suppressWarnings(as.integer(index[sel]))
    if (anyNA(pos) || max(pos) > length(labels[[v]]))
      stop("labels: ", v, " needs one label per element (", sum(sel), ")", call. = FALSE)
    index[sel] <- as.character(labels[[v]])[pos]
  }
  n <- nrow(wide)
  data.frame(
    .chain = rep(wide$.chain, times = length(cols)),
    .iteration = rep(wide$.iteration, times = length(cols)),
    .draw = rep(wide$.draw, times = length(cols)),
    .variable = rep(variable, each = n),
    .index = rep(index, each = n),
    .value = unlist(wide[cols], use.names = FALSE),
    stringsAsFactors = FALSE
  )
}

# as.array(m): iterations x chains x parameters, the layout of the
# bayesplot mcmc_*() functions (and of rstan / posterior draws_array).
# Parameter names are those of bf_draws(): a, b[1], b[2], ...
# Registered in NAMESPACE as S3method(as.array, BayesForge.Main.main.bf, .bf_as_array).
.bf_as_array <- function(x, ...) {
  w <- bf_draws(x)
  vars <- setdiff(names(w), c(".chain", ".iteration", ".draw"))
  C <- max(w$.chain)
  S <- max(w$.iteration)
  out <- array(NA_real_, dim = c(S, C, length(vars)),
               dimnames = list(iteration = NULL, chain = paste0("chain:", seq_len(C)),
                               variable = vars))
  # bf_draws() rows run over chains, then iterations: a column-major S x C fill
  for (k in seq_along(vars)) out[, , k] <- matrix(w[[vars[k]]], nrow = S, ncol = C)
  out
}

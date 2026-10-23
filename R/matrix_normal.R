#' @title Matrix Normal distribution
#'
#' @description
#' Samples from a Matrix Normal distribution, which is a multivariate normal
#' distribution over matrices. The distribution is characterized by a location
#' matrix and two lower triangular matrices that define the correlation
#' structure. The distribution is related to the multivariate normal
#' distribution in the following way. If \eqn{X ~ MN(loc,U,V)} then \eqn{vec(X)
#' ~ MVN(vec(loc), kron(V,U) )}.
#'
#' Let (X) be (n \\times p), with mean matrix
#' \deqn{M\in\mathbb R^{n\times p}}
#' , row-covariance matrix \eqn{U\in\mathbb R^{n\times n}} (positive-definite),
#' and column-covariance matrix \eqn{V\in\mathbb R^{p\times
#' p}}(positive-definite). Then the PDF is:
#'
#' \deqn{f(X;M,U,V) = \frac{1}{(2\pi)^{\tfrac{np}{2}},|V|^{\tfrac n2},|U|^{\tfrac p2}}, \exp!\Big[-\tfrac12,\mathrm{tr}!\big( V^{-1}(X-M)^\top,U^{-1}(X-M)\big)\Big].}
#'
#' Also the equivalence to vec-form is given:
#'
#' \deqn{\mathrm{vec}(X) \sim \mathcal N_{np}\big(\mathrm{vec}(M),,V\otimes U\big). ; ; \text{([The Book of Statistical Proofs][3])}}
#'
#' @param loc See the Python documentation of `loc`.
#' @param scale_tril_row See the Python documentation of `scale_tril_row`.
#' @param scale_tril_column See the Python documentation of `scale_tril_column`.
#' @param validate_args Logical. Whether to validate the parameter values.
#'   Defaults to `py_none()` (the NumPyro default).
#' @param name Character. Name of the random variable in the model. When not
#'   given, an observed site is named after the variable passed to `obs` (`obs =
#'   y` gives `"y"`); use [bf_model()] to name parameters after the variables they
#'   are assigned to.
#' @param obs Numeric vector or array of observed values. If given, the random
#'   variable is conditioned on them (likelihood); otherwise it is latent.
#'   Defaults to `py_none()`.
#' @param mask Optional logical vector or array masking observations (`FALSE` =
#'   ignored). Defaults to `py_none()`.
#' @param sample Logical. If `TRUE`, directly draw values from the distribution
#'   and return them; if `FALSE` (default), create a random variable inside a
#'   model.
#' @param seed Integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param shape Integer vector. With `sample = FALSE`, the batch shape of the
#'   distribution in the model; with `sample = TRUE`, the shape of the draw.
#'   Defaults to `c()` (no extra dimensions).
#' @param event Integer. Number of batch dimensions reinterpreted as event
#'   dimensions (model building only). Defaults to `0`.
#' @param create_obj Logical. If `TRUE`, return the distribution object itself
#'   instead of a random variable (e.g. to build mixtures). Defaults to `FALSE`.
#' @param to_jax Logical. Return a JAX array (`TRUE`, default) or convert to R.
#'
#' @return
#' \itemize{
#'   \item When `sample = FALSE`: A BF MatrixNormal distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     MatrixNormal distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' n_rows <- 3L
#' n_cols <- 4L
#' loc <- jnp$zeros(c(n_rows, n_cols))
#' U_row_cov <- jnp$array(
#'     matrix(c(1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0), nrow = 3L, byrow = TRUE)
#' )
#' V_col_cov <- jnp$array(
#'     matrix(
#'         c(
#'             2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1, 0.2, 2.0, 0.0, 0.4, -0.2, 0.0,
#'             2.0
#'         ),
#'         nrow = 4L,
#'         byrow = TRUE
#'     )
#' )
#' bf.dist.matrix_normal(
#'     loc = loc,
#'     scale_tril_row = jnp$linalg$cholesky(U_row_cov),
#'     scale_tril_column = jnp$linalg$cholesky(V_col_cov),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Matrix_normal_distribution "Matrix
#' normal distribution"
#'
#' \[2\]: https://statproofbook.github.io/P/matn-pdf.html "Probability density
#' function of the matrix-normal distribution | The Book of Statistical Proofs"
#'
#' @export
bf.dist.matrix_normal <- function(loc, scale_tril_row, scale_tril_column, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$matrix_normal(
    loc = .BF_env$jnp$asarray(loc),
    scale_tril_row = .BF_env$jnp$asarray(scale_tril_row),
    scale_tril_column = .BF_env$jnp$asarray(scale_tril_column),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

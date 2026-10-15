#' @title Matrix Normal Distribution
#'
#' @description
#' Samples from a Matrix Normal distribution, which is a multivariate normal distribution over matrices.
#' The distribution is characterized by a location matrix and two lower triangular matrices that define the correlation structure.
#' The distribution is related to the multivariate normal distribution in the following way.
#' If \deqn{X \sim MN(loc,U,V) \implies vec(X) \sim MVN(vec(loc), kron(V,U) )}.
#'
#' @param loc A numeric vector, matrix, or array representing the location of the distribution.
#' @param scale_tril_row A numeric vector, matrix, or array representing the lower cholesky of rows correlation matrix.
#' @param scale_tril_column A numeric vector, matrix, or array representing the lower cholesky of columns correlation matrix.
#' @param shape A numeric vector specifying the shape of the distribution.  Must be a vector.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A logical vector, matrix, or array (.BF_env$jnp$array) to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BF Matrix Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Matrix Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' n_rows <- 3
#' n_cols <- 4
#' loc <- matrix(rep(0, n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE)
#'
#' U_row_cov <-
#'   matrix(c(1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0),
#'     nrow = n_rows, ncol = n_rows, byrow = TRUE
#'   )
#' scale_tril_row <- chol(U_row_cov)
#'
#' V_col_cov <- matrix(
#'   c(
#'     2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1,
#'     0.2, 2.0, 0.0, 0.4, -0.2, 0.0, 2.0
#'   ),
#'   nrow = n_cols, ncol = n_cols, byrow = TRUE
#' )
#' scale_tril_column <- chol(V_col_cov)
#'
#'
#' bf.dist.matrix_normal(
#'   loc = loc,
#'   scale_tril_row = scale_tril_row,
#'   scale_tril_column = scale_tril_column,
#'   sample = TRUE
#' )
#' }
#' @export

bf.dist.matrix_normal <- function(loc, scale_tril_row, scale_tril_column, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
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

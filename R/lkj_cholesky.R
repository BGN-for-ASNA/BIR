#' @title LKJ Cholesky distribution
#'
#' @description
#' The LKJ (Leonard-Kjærgaard-Jørgensen) Cholesky distribution is a family of
#' distributions on symmetric matrices, often used as a prior for the Cholesky
#' decomposition of a symmetric matrix. It is particularly useful in BayesForge
#' for models with covariance structure. Given a lower triangular matrix (L)
#' with unit diagonal entries, the PDF is:
#'
#' \deqn{p(L | \eta) \propto \prod_{k=2}^d L_{kk}^{d-k+2\eta-2}}
#'
#' where:
#' * (L_\{kk\}) is the diagonal element of (L).
#' * (\\eta) is the shape parameter.
#' * (d) is the dimension of the matrix.
#'
#' * (\\eta = 1): Uniform prior over correlation matrices.
#' * (\\eta > 1): Prior favors correlation matrices close to the identity matrix
#' (i.e., weak correlations).
#' * (\\eta < 1): Prior allows stronger correlations.
#'
#' @param dimension See the Python documentation of `dimension`.
#' @param concentration See the Python documentation of `concentration`.
#' @param sample_method See the Python documentation of `sample_method`.
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
#' @return A random variable (model building) or draws (`sample = TRUE`).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.lkj_cholesky(dimension = 3L, concentration = 2.0, sample = TRUE)
#' }
#'
#' @references
#' https://mc-stan.org/docs/2_21/functions-reference/cholesky-lkj-correlation-distribution.html
#'
#' @export
bf.dist.lkj_cholesky <- function(dimension, concentration=1.0, sample_method='onion', validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  dimension <- as.integer(dimension)
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
  .BF_env$.bf_instance$dist$lkj_cholesky(
    dimension = .BF_env$jnp$asarray(dimension),
    concentration = .BF_env$jnp$asarray(concentration),
    sample_method = sample_method,
    validate_args = validate_args,
    name = name,
    obs = obs,
    mask = mask,
    sample = sample,
    seed = seed,
    shape = shape,
    event = event,
    create_obj = create_obj
  )
}

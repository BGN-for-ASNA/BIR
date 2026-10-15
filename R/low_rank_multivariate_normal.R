#' @title Low Rank Multivariate Normal distribution
#'
#' @description
#' The *Low-Rank Multivariate Normal* (LRMVN) distribution is a parameterization
#' of the multivariate normal distribution where the covariance matrix is
#' expressed as a low-rank plus diagonal decomposition:
#'
#' \deqn{\Sigma = F F^\top + D}
#'
#' where \eqn{F} is a low-rank matrix (capturing correlations) and \eqn{D} is a
#' diagonal matrix (capturing independent noise). This representation is often
#' used in probabilistic modeling and variational inference to efficiently
#' handle high-dimensional Gaussian distributions with structured covariance.
#'
#' For a (d)-dimensional random vector ( x \\in \\mathbb\{R\}^d ),
#'
#' \deqn{p(x \mid \mu, F, D) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right),}
#'
#' where ( \\Sigma = F F^\\top + D ).
#'
#' @param loc Mean vector. cov_factor (numeric vector or array): Matrix used to
#'   construct the covariance matrix. cov_diag (numeric vector or array): Diagonal
#'   elements of the covariance matrix.
#' @param cov_factor See the Python documentation of `cov_factor`.
#' @param cov_diag See the Python documentation of `cov_diag`.
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
#' event_size <- 100L
#' rank <- 5L
#' bf.dist.low_rank_multivariate_normal(loc = bf.dist.normal(0, 1, shape = c(event_size), sample = TRUE) * 2, cov_factor = bf.dist.normal(0, 1, shape = c(event_size, rank), sample = TRUE), cov_diag = jnp$exp(bf.dist.normal(0, 1, shape = c(event_size), sample = TRUE)) * 0.1, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#lowrankmultivariatenormal}
#'
#' @references
#' [Multivariate normal distribution –
#' Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
#'
#' [TensorFlow Probability:
#' LowRankMultivariateNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRank)
#'
#' @export
bf.dist.low_rank_multivariate_normal=function(loc, cov_factor, cov_diag, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$low_rank_multivariate_normal(
       loc = .BF_env$jnp$asarray(loc),
       cov_factor = .BF_env$jnp$asarray(cov_factor),
       cov_diag = .BF_env$jnp$asarray(cov_diag),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

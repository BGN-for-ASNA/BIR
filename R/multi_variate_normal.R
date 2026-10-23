#' @title Multivariate Normal distribution
#'
#' @description
#' The Multivariate Normal distribution, also known as the Gaussian distribution
#' in multiple dimensions, is a probability distribution that arises frequently
#' in statistics and machine learning. It is defined by its mean vector and
#' covariance matrix, which describe the central tendency and spread of the
#' distribution, respectively.
#'
#' \deqn{p(x) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)}
#'
#' where:
#' - \eqn{x} is a \eqn{n}-dimensional vector of random variables.
#' - \eqn{\mu} is the mean vector.
#' - \eqn{\Sigma} is the covariance matrix.
#'
#' @param loc The mean vector of the distribution.
#' @param covariance_matrix The covariance matrix of the distribution. Must be
#'   positive definite.
#' @param precision_matrix The precision matrix (inverse of the covariance
#'   matrix) of the distribution. Must be positive definite.
#' @param scale_tril The lower triangular Cholesky decomposition of the
#'   covariance matrix. - `shape` (integer vector): A multi-purpose argument for
#'   shaping. When `sample = FALSE` (model building), this is used with `shape` to
#'   set the distribution's batch shape. When `sample = TRUE` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the given shape.
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
#'   \item When `sample = FALSE`: A BF MultivariateNormal distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     MultivariateNormal distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.multivariate_normal(
#'     loc = jnp$array(c(1.0, 0.0, -2.0)),
#'     covariance_matrix = jnp$array(
#'         matrix(c(2.0, 0.7, -0.3, 0.7, 1.0, 0.5, -0.3, 0.5, 1.5), nrow = 3L, byrow = TRUE)
#'     ),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multivariate-normal}
#'
#' @references
#' [Multivariate normal distribution –
#' Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
#'
#' @export
bf.dist.multivariate_normal=function(loc=0.0, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if(!.BF_env$.py$is_none(covariance_matrix)){covariance_matrix = .BF_env$jnp$asarray(covariance_matrix)}
     if(!.BF_env$.py$is_none(precision_matrix)){precision_matrix = .BF_env$jnp$asarray(precision_matrix)}
     if(!.BF_env$.py$is_none(scale_tril)){scale_tril = .BF_env$jnp$asarray(scale_tril)}

     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     .BF_env$.bf_instance$dist$multivariate_normal(
       loc = .BF_env$jnp$asarray(loc),
       covariance_matrix = covariance_matrix,
       precision_matrix = precision_matrix,
       scale_tril = scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

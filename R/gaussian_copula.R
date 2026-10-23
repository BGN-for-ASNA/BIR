#' @title Gaussian Copula distribution
#'
#' @description
#' A distribution that links the `batch_shape\[:-1\]` of a marginal distribution
#' with a multivariate Gaussian copula, modelling the correlation between the
#' axes. A copula is a multivariate distribution over the uniform distribution
#' on \[0, 1\]. The Gaussian copula links the marginal distributions through a
#' multivariate normal distribution.
#'
#' @param marginal_dist Distribution whose last batch axis is to be coupled.
#' @param correlation_matrix Correlation matrix of the coupling multivariate
#'   normal distribution. Defaults to py_none().
#' @param correlation_cholesky Correlation Cholesky factor of the coupling
#'   multivariate normal distribution. Defaults to py_none().
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
#'   \item BF GaussianCopula distribution object: When `sample = FALSE` (for model
#'     building).
#'   \item JAX array: When `sample = TRUE` (for direct sampling).
#'   \item BF distribution object: When `create_obj = TRUE` (for advanced use
#'     cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.gaussian_copula(
#'     marginal_dist = bf.dist.beta(2.0, 5.0, create_obj = TRUE),
#'     correlation_matrix = jnp$array(
#'         matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2L, byrow = TRUE)
#'     ),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#gaussiancopula}
#'
#' @export
bf.dist.gaussian_copula=function(marginal_dist, correlation_matrix=py_none(), correlation_cholesky=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"


      return(print("No more available since jax > 0.06"))

      shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(correlation_cholesky)){correlation_cholesky = .BF_env$jnp$asarray(correlation_cholesky)}
     if(!.BF_env$.py$is_none(correlation_matrix)){correlation_matrix = .BF_env$jnp$asarray(correlation_matrix)}
     .BF_env$.bf_instance$dist$gaussian_copula(
       marginal_dist = marginal_dist,
       correlation_matrix = correlation_matrix,
       correlation_cholesky = correlation_cholesky,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

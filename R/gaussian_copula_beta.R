#' @title Gaussian Copula Beta distribution
#'
#' @description
#' This distribution combines a Gaussian copula with a Beta distribution. The
#' Gaussian copula models the dependence structure between random variables,
#' while the Beta distribution defines the marginal distributions of each
#' variable.
#'
#' @param concentration1 The first shape parameter of the Beta distribution.
#' @param concentration0 The second shape parameter of the Beta distribution.
#' @param correlation_matrix Correlation matrix of the coupling multivariate
#'   normal distribution. Defaults to py_none().
#' @param correlation_cholesky The Cholesky decomposition of the correlation
#'   matrix.
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
#'   \item GaussianCopulaBeta: A BF GaussianCopulaBeta distribution object (for
#'     model building).
#'   \item numeric vector or array: A JAX array of samples drawn from the
#'     GaussianCopulaBeta distribution (for direct sampling).
#'   \item Distribution: The raw BF distribution object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.gaussian_copula_beta(
#'     concentration1 = jnp$array(c(2.0, 3.0)),
#'     concentration0 = jnp$array(c(5.0, 3.0)),
#'     correlation_cholesky = jnp$linalg$cholesky(
#'         jnp$array(matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2L, byrow = TRUE))
#'     ),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#gaussiancopulabetadistribution}
#'
#' @export
bf.dist.gaussian_copula_beta=function(concentration1, concentration0, correlation_matrix=py_none(), correlation_cholesky=py_none(), validate_args=FALSE, name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     return(print("No more available since jax > 0.06"))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(correlation_cholesky)){correlation_cholesky = .BF_env$jnp$asarray(correlation_cholesky)}
     if(!.BF_env$.py$is_none(correlation_matrix)){correlation_matrix = .BF_env$jnp$asarray(correlation_matrix)}

     .BF_env$.bf_instance$dist$gaussian_copula_beta(
       concentration1 = .BF_env$jnp$asarray(concentration1),
       concentration0 = .BF_env$jnp$asarray(concentration0),
       correlation_matrix = correlation_matrix,
       correlation_cholesky = correlation_cholesky,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

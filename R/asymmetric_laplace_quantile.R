#' @title Asymmetric Laplace Quantile Distribution
#'
#' @description
#' Samples from an Asymmetric Laplace Quantile distribution.
#' This distribution is an alternative parameterization of the Asymmetric Laplace
#' distribution, commonly used in Bayesian quantile regression. It utilizes a
#' `quantile` parameter to define the balance between the left- and right-hand
#' sides of the distribution, representing the proportion of probability density
#' that falls to the left-hand side.
#'
#' @param loc The location parameter of the distribution.
#'
#' @param scale The scale parameter of the distribution.
#'
#' @param quantile The quantile parameter, representing the proportion of
#'   probability density to the left of the median. Must be between 0 and 1.
#'
#' @param shape A numeric vector. When `sample=False` (model building), this is
#'   used with `.expand(shape)` to set the distribution's batch shape. When
#'   `sample=True` (direct sampling), this is used as `sample_shape` to draw a
#'   raw JAX array of the given shape.
#'
#' @param event The number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#'
#' @param mask An optional boolean array to mask observations.
#'
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#'
#' @param create_obj If `TRUE`, returns the raw NumPyro distribution object
#'   instead of creating a sample site. This is essential for building complex
#'   distributions like `MixtureSameFamily`.
#'
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#'
#' @param seed An integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#'
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#'
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BF Asymmetric Laplace Quantile distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Asymmetric Laplace Quantile distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).


#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplacequantile}


#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.asymmetric_laplace_quantile(sample = TRUE)
#' }
#' @export
bf.dist.asymmetric_laplace_quantile=function(loc=0.0, scale=1.0, quantile=0.5, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$asymmetric_laplace_quantile(
       loc=.BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       quantile= .BF_env$jnp$asarray(quantile),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

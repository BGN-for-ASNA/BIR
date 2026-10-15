#' @title Uniform Distribution
#'
#' @description
#' The Uniform distribution is the simplest continuous distribution: every value in the interval ([a, b]) is **equally likely**.
#'  It is widely used for modeling complete randomness within a fixed range, random sampling, and as a building block for other distributions.
#'
#' @param low A numeric vector, matrix, or array representing the lower bound of the uniform interval.
#' @param high A numeric vector, matrix, or array representing the upper bound of the uniform interval.
#' @param shape A numeric vector specifying the shape of the output. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions
#'   (used in model building).
#' @param mask A logical vector, matrix, or array (optional) to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object
#'   instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BF Uniform distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Uniform distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.uniform(low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bf.dist.uniform=function(low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$uniform(
       low = .BF_env$jnp$asarray(low),
       high = .BF_env$jnp$asarray(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

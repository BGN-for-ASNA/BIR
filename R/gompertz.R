#' @title Gompertz Distribution
#'
#' @description The Gompertz distribution is a distribution with support on the positive real line that is closely
#' related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
#' entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.

#' @param concentration A positive numeric vector, matrix, or array representing the concentration parameter.
#' @param rate A positive numeric vector, matrix, or array representing the rate parameter.
#' @param shape A numeric vector representing the shape parameter.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A boolean vector, matrix, or array representing an optional mask for observations.
#' @param create_obj Logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample site.
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
#'    A BF Gompertz distribution object when \code{sample=FALSE} (for model building).
#'
#'    A JAX array when \code{sample=TRUE} (for direct sampling).
#'
#'    A BF distribution object when \code{create_obj=TRUE} (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#gompertz}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.gompertz(concentration = 5., sample = TRUE)
#' }
#' @export
bf.dist.gompertz=function(concentration, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$gompertz(
       concentration = .BF_env$jnp$asarray(concentration),
       rate = .BF_env$jnp$asarray(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

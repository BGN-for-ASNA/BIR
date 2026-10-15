#' @title Samples from a Chi-squared distribution.
#'
#' @description
#' The Chi-squared distribution is a continuous probability distribution that arises
#'  frequently in hypothesis testing, particularly in ANOVA and chi-squared tests.
#'  It is defined by a single positive parameter, degrees of freedom (df), the number of independent standard normal variables squared and summed, which
#'  determines the shape of the distribution.

#' @param df A numeric vector representing the degrees of freedom. Must be positive.
#' @param shape A numeric vector used for shaping. When `sample=FALSE` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
#'   of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object
#'   instead of creating a sample site. This is essential for building complex distributions
#'   like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BF Chi-squared distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Chi-squared distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#chi2}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.chi2(c(0,2),sample = TRUE)
#' }
#' @export
#'
bf.dist.chi2=function(df, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$chi2(
       df = .BF_env$jnp$asarray(df),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

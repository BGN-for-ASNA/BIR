#' @title Two-Sided Truncated Distribution
#' @description
#' A "two-sided truncated distribution" is a general concept: you take a base continuous distribution
#' and **restrict it** to an interval ([`low`, `high`]), discarding all mass outside, then **renormalize**
#'  so the inner portion integrates to 1. I'll spell out the general formulas, caveats, sampling strategies,
#'  and special cases (e.g. truncated normal) to illustrate.
#'
#' @param base_dist The base distribution to truncate.
#' @param low The lower bound for truncation.
#' @param high The upper bound for truncation.
#' @param sample Logical; if `TRUE`, returns JAX array of samples.  Defaults to `FALSE`.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector to mask observations.
#' @param create_obj Logical; if `TRUE`, returns the raw BF distribution object. Defaults to `FALSE`.
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
#'  - When \code{sample=FALSE}, a BF Two-Sided Truncated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Two-Sided Truncated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.two_sided_truncated_distribution(
#' base_dist = bf.dist.normal(0,1, create_obj = TRUE),
#' high = 0.5, low = 0.1, sample = TRUE)
#' }
#' @export
bf.dist.two_sided_truncated_distribution=function(base_dist, low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$two_sided_truncated_distribution(
       base_dist,
       low= .BF_env$jnp$asarray(low),
       high = .BF_env$jnp$asarray(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

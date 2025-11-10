#' @title Samples from a Gumbel (or Extreme Value) distribution.
#'
#' @description The Gumbel distribution is a continuous probability distribution named after German mathematician Carl Gumbel.
#' It is often used to model the distribution of maximum values in a sequence of independent random variables.
#'
#' \deqn{f(x) = \frac{1}{s} e^{-(x - \mu) / s} e^{-e^{- (x - \mu) / s}}}
#'
#' @param loc Location parameter.
#' @param scale Scale parameter. Must be positive.
#' @param shape A numeric vector. When \code{sample=FALSE} (model building), this is used with \code{.expand(shape)} to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as \code{sample_shape} to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Optional boolean array to mask observations.
#' @param create_obj If \code{TRUE}, returns the raw BI distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Gumbel distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Gumbel distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.gumbel(sample = TRUE)
#' }
#' @export
bi.dist.gumbel=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$gumbel(
       loc=.BI_env$jnp$array(loc),
       scale= .BI_env$jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title BetaBinomial
#' @description Samples from a Beta-Binomial distribution.
#' @param concentration1 A numeric vector, matrix, or array representing the first concentration parameter (alpha) of the Beta distribution.
#' @param concentration0 A numeric vector, matrix, or array representing the second concentration parameter (beta) of the Beta distribution.
#' @param total_count A numeric vector, matrix, or array representing the number of Bernoulli trials in the Binomial part of the distribution.
#' @param shape A numeric vector.  When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Beta-Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Beta-Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).

#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#betabinomial}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.beta_binomial(0,1,sample = TRUE)
#' }
#' @export
bi.dist.beta_binomial=function(concentration1, concentration0, total_count=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$beta_binomial(
       concentration1 = jnp$array(concentration1),
       concentration0 = jnp$array(concentration0),
       total_count = jnp$array(as.integer(total_count)),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}



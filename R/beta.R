#' @description Samples from a Beta distribution, defined on the interval [0, 1].
#' The Beta distribution is a versatile distribution often used to model
#' probabilities or proportions. It is parameterized by two positive shape
#' parameters, often referred to as concentration parameters in the BI
#' context.
#'
#' @title Beta Distribution
#'
#' @param concentration1 A numeric vector or array representing the first concentration parameter (shape parameter). Must be positive.
#' @param concentration0 A numeric vector or array representing the second concentration parameter (shape parameter). Must be positive.
#' @param shape A numeric vector.  When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample
#'   site. This is essential for building complex distributions like `MixtureSameFamily`.

#' @return
#'  - When \code{sample=FALSE}, a BI Beta distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Beta distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'

#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#beta}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.beta(concentration1 = 0, concentration0 = 1, sample = TRUE)
#' }
#' @export
bi.dist.beta=function(concentration1, concentration0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$beta(
       concentration1 = jnp$array(concentration1),
       concentration0 = jnp$array(concentration0),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

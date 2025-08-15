#' @title Samples from a Logistic distribution.
#' @description
#' The Logistic distribution is a continuous probability distribution defined by two parameters: location and scale. It is often used to model growth processes and is closely related to the normal distribution.
#' @param loc Numeric vector or single number. The location parameter, specifying the median of the distribution. Defaults to 0.0.
#' @param scale Numeric vector or single number. The scale parameter, which determines the spread of the distribution. Must be positive. Defaults to 1.0.
#' @param shape Numeric vector. A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @return
#'  - When \code{sample=FALSE}, a BI Logistic distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Logistic distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.logistic(sample = TRUE)
#' }
#' @export
bi.dist.logistic=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$logistic(
       loc = jnp$array(loc),
       scale = jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

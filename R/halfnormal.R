#' @title Samples from a HalfNormal distribution.
#'
#' @description
#' The HalfNormal distribution is a distribution of the absolute value of a normal random variable.
#' It is defined by a location parameter (implicitly 0) and a scale parameter.
#'
#' @param scale A numeric vector or array representing the scale parameter of the distribution. Must be positive.
#' @param shape A numeric vector used for shaping. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array representing an optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a
#'   sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI HalfNormal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the HalfNormal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.half_normal(sample = TRUE)
#' }
#' @export

bi.dist.half_normal=function(scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$half_normal(
       scale=jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

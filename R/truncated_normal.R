#' @title Truncated Normal Distribution
#' @description The Truncated Normal distribution is a normal distribution truncated
#' to a specified interval. It is defined by its location (`loc`), scale
#' (`scale`), lower bound (`low`), and upper bound (`high`).
#'
#' @param loc The location parameter of the normal distribution.
#' @param scale The scale parameter of the normal distribution.
#' @param shape A numeric vector (e.g., `c(10)`) used to shape the distribution.
#'   When `sample=False` (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the
#'   given shape.
#' @param event The number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution
#'   object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Truncated Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_normal(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bi.dist.truncated_normal=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$truncated_normal(
       loc=jnp$array(loc),
       scale= jnp$array(scale),
       low= jnp$array(low),
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Truncated Cauchy Distribution
#' @description
#' Samples from a Truncated Cauchy distribution.
#'
#' The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
#' that appears frequently in various areas of mathematics and physics. It is characterized by its heavy tails,
#' which extend to infinity. The truncated version limits the support of the Cauchy distribution to a specified interval.
#'
#' \deqn{f(x) = \frac{1}{\pi \cdot c \cdot (1 + ((x - b) / c)^2)}  \text{ for } a < x < b}
#'
#' @param loc Location parameter of the Cauchy distribution.
#' @param scale Scale parameter of the Cauchy distribution.
#' @param shape A numeric vector. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Truncated Cauchy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated Cauchy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_cauchy(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bi.dist.truncated_cauchy=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$truncated_cauchy(
       loc=jnp$array(loc),
       scale= jnp$array(scale),
       low= jnp$array(low),
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

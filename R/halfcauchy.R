#' @title HalfCauchy Distribution
#'
#' @description
#' The HalfCauchy distribution is a probability distribution that is half of the Cauchy distribution.
#' It is defined on the positive real numbers and is often used in situations where only positive values are relevant.
#'
#' \deqn{ f(x) = \frac{1}{2} \cdot \frac{1}{\pi \cdot \frac{1}{scale} \cdot (x^2 + \frac{1}{scale^2})}}
#'
#' @param scale A numeric vector representing the scale parameter of the Cauchy distribution. Must be positive.
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, optionally used to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI HalfCauchy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the HalfCauchy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#halfcauchy}
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.halfcauchy(sample = TRUE)
#' }
#' @export
bi.dist.half_cauchy=function(scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$half_cauchy(
       scale=jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

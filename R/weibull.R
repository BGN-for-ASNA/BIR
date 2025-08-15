#' @title Samples from a Weibull distribution.
#' @description
#' The Weibull distribution is a versatile distribution often used to model failure rates in engineering and reliability studies. It is characterized by its shape and scale parameters.
#'
#' \deqn{f(x) = \frac{\beta}{\alpha} \left(\frac{x}{\alpha}\right)^{\beta - 1} e^{-\left(\frac{x}{\alpha}\right)^{\beta}} \text{ for } x \ge 0}
#'
#' where \deqn{\alpha} is the scale parameter and \eqn{\beta} is the shape parameter.
#'
#' @param scale A numeric vector, matrix, or array representing the scale parameter of the Weibull distribution. Must be positive.
#' @param concentration A numeric vector, matrix, or array representing the shape parameter of the Weibull distribution. Must be positive.
#' @param shape A numeric vector.  This is used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Weibull distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Weibull distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#weibull}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.weibull(scale = c(10, 10), concentration = c(1,1), sample = TRUE)
#' }
#' @export
bi.dist.weibull=function(scale, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$weibull(
       scale = jnp$array(scale),
       concentration = jnp$array(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

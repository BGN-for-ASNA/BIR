#' @title  Levy distribution.
#'
#' @description Samples from a Levy distribution.
#'
#' The probability density function is given by,
#'
#' \deqn{f(x\mid \mu, c) = \sqrt{\frac{c}{2\pi(x-\mu)^{3}}} \exp\left(-\frac{c}{2(x-\mu)}\right), \qquad x > \mu}
#'
#' where \deqn{\mu} is the location parameter and \deqn{c} is the scale parameter.
#'
#' @param loc A numeric vector, matrix, or array representing the location parameter.
#' @param scale A numeric vector, matrix, or array representing the scale parameter.
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Levy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Levy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#levy}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.levy( loc = 1, scale = 10,  sample = TRUE)
#' }
#' @export

bi.dist.levy=function(loc, scale, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$levy(
       loc = jnp$array(loc),
       scale = jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

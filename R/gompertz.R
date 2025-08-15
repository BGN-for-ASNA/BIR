#' @title Gompertz Distribution
#'
#' @description The Gompertz distribution is a distribution with support on the positive real line that is closely
#' related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
#' entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.
#'
#' The probability density function (PDF) is:
#'
#' \deqn{f(x) = \frac{con}{rate} \exp \left\{ - \frac{con}{rate} \left [ \exp\{x * rate \} - 1 \right ] \right\} \exp(-x * rate)}
#'
#' @param concentration A positive numeric vector, matrix, or array representing the concentration parameter.
#' @param rate A positive numeric vector, matrix, or array representing the rate parameter.
#' @param shape A numeric vector representing the shape parameter.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A boolean vector, matrix, or array representing an optional mask for observations.
#' @param create_obj Logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'    A BI Gompertz distribution object when \code{sample=FALSE} (for model building).
#'
#'    A JAX array when \code{sample=TRUE} (for direct sampling).
#'
#'    A BI distribution object when \code{create_obj=TRUE} (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#gompertz}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.gompertz(concentration = 5., sample = TRUE)
#' }
#' @export
bi.dist.gompertz=function(concentration, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$gompertz(
       concentration = jnp$array(concentration),
       rate = jnp$array(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

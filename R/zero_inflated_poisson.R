#' @title A Zero Inflated Poisson distribution.
#' @description
#' This distribution combines two Poisson processes: one with a rate parameter and another that generates only zeros.
#' The probability of observing a zero is determined by the 'gate' parameter, while the probability of observing a non-zero value is governed by the 'rate' parameter of the underlying Poisson distribution.
#'
#' \deqn{P(X = k) = (1 - gate) * \frac{e^{-rate} rate^k}{k!} + gate}
#'
#' @param gate The gate parameter.
#' @param rate A numeric vector, matrix, or array representing the rate parameter of the underlying Poisson distribution.
#' @param shape A numeric vector used to shape the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector, matrix, or array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @param sample Logical; If `TRUE`, draws samples from the distribution.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Zero Inflated Poisson distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero Inflated Poisson distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' #' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.zero_inflated_poisson(gate=0.3, rate = 5, sample = TRUE)
#' }
#' @export
bi.dist.zero_inflated_poisson=function(gate, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$zero_inflated_poisson(
       gate = jnp$array(gate),
       rate = jnp$array(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Poisson Distribution
#'
#' @description
#' A discrete probability distribution that models the number of events occurring in a fixed interval of time or space if these events occur with a known average rate and independently of the time since the last event.
#' Creates a Poisson distribution, a discrete probability distribution that models the number of events occurring in a fixed interval of time or space if these events occur with a known average rate and independently of the time since the last event.
#'
#' \deqn{ \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}}
#'
#' @param rate A numeric vector representing the average number of events.
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Poisson distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Poisson distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.poisson(rate = c(0.2, 0.5, 0.8), sample = TRUE)
#' }
#' @export
bi.dist.poisson=function(rate, is_sparse=FALSE, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$poisson(
       rate = jnp$array(rate),
       is_sparse= is_sparse,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

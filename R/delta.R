#' @title The Delta distribution.
#'
#' @description The Delta distribution, also known as a point mass distribution, assigns probability 1 to a single point and 0 elsewhere. It's useful for representing deterministic variables or as a building block for more complex distributions.
#'
#' \deqn{P(x = v) = 1}
#'
#' @importFrom reticulate py_none tuple
#' @param v A numeric vector representing the location of the point mass.
#' @param shape A numeric vector used for shaping. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Delta distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Delta distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#delta}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.delta(v = 5, sample = TRUE)
#' }
#' @export
#'
bi.dist.delta=function(v=0.0, log_density=0.0, event_dim=0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event_dim=as.integer(event)
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$delta(
       v=jnp$array(v),
       log_density= jnp$array(log_density),
       event_dim= jnp$array(event_dim),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

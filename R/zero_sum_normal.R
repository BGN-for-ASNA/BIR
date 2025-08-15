#' @title zero_sum_normal
#'
#' @description
#' Samples from a zero_sum_normal distribution, which is a Normal distribution where one or more axes are constrained to sum to zero.
#'
#' @param scale A numeric vector or array representing the standard deviation of the underlying normal distribution before the zerosum constraint is enforced.
#' @param shape A numeric vector specifying the shape of the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI zero_sum_normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the MultinomialProbs distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zerosumnormal}
#'
#' @examples
#' #' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.zero_sum_normal(scale=0.3, event_shape = c(), sample = TRUE)
#' }
#' @export
bi.dist.zero_sum_normal=function(scale, event_shape, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event_shape=do.call(tuple, as.list(as.integer(event_shape)))
     seed=as.integer(seed);
     .bi$dist$zero_sum_normal(
       scale = jnp$array(scale),
       event_shape = event_shape,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

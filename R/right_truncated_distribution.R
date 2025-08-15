#' @title Samples from a right-truncated distribution.
#' @description
#' This distribution truncates the base distribution at a specified high value.  Values greater than `high` are discarded,
#' effectively creating a distribution that is only supported up to that point. This is useful for modeling data
#' where observations are only possible within a certain range.
#'
#' \deqn{f(x) = \frac{f(x)}{P(X \le high)}}
#'
#' @param base_dist The base distribution to truncate. Must be a univariate distribution with real support.
#' @param shape A numeric vector. When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI right-truncated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the right-truncated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.right_truncated_distribution(
#' base_dist = bi.dist.normal(0,1, create_obj = TRUE),
#' high = 10,
#' sample = TRUE)
#' }
#' @export
bi.dist.right_truncated_distribution=function(base_dist, high=0.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$right_truncated_distribution(
       base_dist,
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

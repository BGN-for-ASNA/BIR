#' @title Truncated Distribution
#'
#' @description
#' Samples from a Truncated Distribution.
#' This distribution represents a base distribution truncated between specified lower and upper bounds.
#' The truncation modifies the probability density function (PDF) of the base distribution,
#' effectively removing observations outside the defined interval.
#'
#' \deqn{p(x) = \frac{p(x)}{P(\text{lower} \le x \le \text{upper})}}
#'
#' @param base_dist The base distribution to be truncated. This should be a univariate
#'   distribution. Currently, only the following distributions are supported:
#'   Cauchy, Laplace, Logistic, Normal, and StudentT.
#' @param shape A numeric vector (e.g., `c(10)`) specifying the shape. When \code{sample=FALSE}
#'   (model building), this is used with `.expand(shape)` to set the distribution's
#'   batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape`
#'   to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions
#'   (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site. This is essential for building complex distributions
#'   like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Truncated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#truncateddistribution}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_distribution(
#' base_dist = bi.dist.normal(0,1, create_obj = TRUE),
#' high = 0.7,
#' low = 0.1,
#' sample = TRUE)
#' }
#' @export
bi.dist.truncated_distribution=function(base_dist, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$truncated_distribution(
       base_dist,
       low = jnp$array(low),
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

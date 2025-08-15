#' @title Samples from a Discrete Uniform distribution.
#'
#' @description The Discrete Uniform distribution defines a uniform distribution over a range of integers.
#' It is characterized by a lower bound (`low`) and an upper bound (`high`), inclusive.
#'
#' \deqn{P(X = k) = \frac{1}{high - low + 1}, \quad k \in \{low, low+1, ..., high\}}
#'
#' @title DiscreteUniform
#' @description Samples from a Discrete Uniform distribution.
#' @param low A numeric vector representing the lower bound of the uniform range, inclusive.
#' @param high A numeric vector representing the upper bound of the uniform range, inclusive.
#' @param shape A numeric vector.  When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If TRUE, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @return
#'  - When \code{sample=FALSE}, a BI Discrete Uniform distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Discrete Uniform distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.discrete_uniform(sample = TRUE)
#' }
#' @export
bi.dist.discrete_uniform=function(low=0, high=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$discrete_uniform(
       low = jnp$array(low),
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

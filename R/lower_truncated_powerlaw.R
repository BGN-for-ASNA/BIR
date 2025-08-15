#' @title Lower Truncated Power Law Distribution
#'
#' @description Lower truncated power law distribution with `alpha` index.
#'
#' The probability density function (PDF) is given by:
#'
#' \deqn{f(x; \alpha, a) = (-\alpha-1)a^{-\alpha - 1}x^{-\alpha}, \qquad x \geq a, \qquad \alpha < -1,}
#'
#' where `a` is the lower bound.
#'
#' @param alpha A numeric vector: index of the power law distribution. Must be less than -1.
#' @param low A numeric vector: lower bound of the distribution. Must be greater than 0.
#' @param shape A numeric vector: A multi-purpose argument for shaping. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer: The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector: Optional boolean array to mask observations.
#' @param create_obj Logical: If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Lower Truncated Power Law distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Lower Truncated Power Law distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lowertruncatedpowerlaw}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.lower_truncated_power_law( alpha = c(-2, 2), low = c(1, 0.5),  sample = TRUE)
#' }
#' @export

bi.dist.lower_truncated_power_law=function(alpha, low, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$lower_truncated_power_law(
       alpha = jnp$array(alpha),
       low = jnp$array(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

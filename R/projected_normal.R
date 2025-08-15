#' @title Samples from a Projected Normal distribution.
#' @description
#' This distribution over directional data is qualitatively similar to the von
#' Mises and von Mises-Fisher distributions, but permits tractable variational
#' inference via reparametrized gradients.
#' \deqn{p(x) = \frac{1}{Z} \exp\left(-\frac{1}{2\sigma^2} ||x - \mu||^2\right)}
#'
#' @param concentration A numeric vector representing the concentration parameter,
#'   representing the direction towards which the samples are concentrated.
#' @param shape A numeric vector used for shaping. When \code{sample=FALSE} (model
#'   building), this is used with `.expand(shape)` to set the distribution's
#'   batch shape. When \code{sample=TRUE} (direct sampling), this is used as
#'   `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution
#'   object instead of creating a sample site. This is essential for building
#'   complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Projected Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Projected Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#projectednormal}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.projected_normal(concentration = c(1.0, 3.0, 2.0), sample = TRUE)
#' }
#' @export

bi.dist.projected_normal=function(concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$projected_normal(
       concentration = jnp$array(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

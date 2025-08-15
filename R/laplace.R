#' @title Laplace Distribution
#'
#' @description Samples from a Laplace distribution, also known as the double exponential distribution.
#' The Laplace distribution is defined by its location parameter (loc) and scale parameter (scale).
#'
#' \deqn{f(x) = \frac{1}{2s} \exp\left(-\frac{|x - \mu|}{s}\right)}
#'
#' @param loc A numeric vector representing the location parameter of the Laplace distribution.
#' @param scale A numeric vector representing the scale parameter of the Laplace distribution. Must be positive.
#' @param shape A numeric vector used for shaping. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, optionally used to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a
#'   sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'    - When \code{sample=FALSE}: A BI Laplace distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the Laplace distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#laplace}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.laplace(sample = TRUE)
#' }
#' @export
bi.dist.laplace=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$laplace(
       loc = jnp$array(loc),
       scale = jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

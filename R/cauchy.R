#' @title Cauchy Distribution
#'
#' @description Samples from a Cauchy distribution.
#'
#' The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
#' that arises frequently in various fields, including physics and statistics. It is characterized by its
#' heavy tails, which extend indefinitely.
#'
#' \deqn{f(x) = \frac{1}{\pi \gamma} \left[ \frac{\gamma^2}{(x - \mu)^2 + \gamma^2} \right]}
#'
#' @param loc A numeric vector or scalar representing the location parameter. Defaults to 0.0.
#' @param scale A numeric vector or scalar representing the scale parameter. Must be positive. Defaults to 1.0.
#' @param shape A numeric vector specifying the shape of the distribution.  When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#'   Defaults to `reticulate::py_none()`.
#' @param mask A logical vector, optional, to mask observations. Defaults to `reticulate::py_none()`.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample
#'   site. Defaults to `FALSE`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Cauchy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Cauchy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#cauchy}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.cauchy(sample = TRUE)
#' }
#' @export
bi.dist.cauchy=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$cauchy(
       loc=jnp$array(loc),
       scale= jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title InverseGamma Distribution
#'
#' @description The InverseGamma distribution is a two-parameter family of continuous probability
#' distributions. It is defined by its shape and rate parameters. It is often used as a prior distribution for
#' precision parameters (inverse variance) in Bayesian statistics.
#'
#' \deqn{p(x) = \frac{1}{Gamma(\alpha)} \left( \frac{\beta}{\Gamma(\alpha)} \right)^{\alpha} x^{\alpha - 1} e^{-\beta x}
#' \text{ for } x > 0}
#'
#' @param concentration A numeric vector representing the shape parameter (\\alpha) of the InverseGamma distribution. Must be positive.
#' @param rate A numeric vector representing the rate parameter (\\beta) of the InverseGamma distribution. Must be positive.
#' @param shape A numeric vector. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'   - When \code{sample=FALSE}: A BI InverseGamma distribution object (for model building).
#'
#'   - When \code{sample=TRUE}: A JAX array of samples drawn from the InverseGamma distribution (for direct sampling).
#'
#'   - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#inversegamma}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.inverse_gamma(concentration = 5., sample = TRUE)
#' }
#' @export

bi.dist.inverse_gamma=function(concentration, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$inverse_gamma(
       concentration = jnp$array(concentration),
       rate = jnp$array(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

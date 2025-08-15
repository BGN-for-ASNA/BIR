#' @title Log Normal distribution
#' @description The Log Normal distribution is a probability distribution defined for positive real-valued random variables,
#' parameterized by a location parameter (loc) and a scale parameter (scale).  It arises when the logarithm
#' of a random variable is normally distributed.
#'
#' \deqn{f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(log(x) - \mu)^2}{2\sigma^2}}}
#'
#' @param loc Numeric; Location parameter.
#' @param scale Numeric; Scale parameter.
#' @param shape Numeric vector; A multi-purpose argument for shaping. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
#'   of the given shape.
#' @param event Numeric; The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector; Optional boolean array to mask observations.
#' @param create_obj Logical; If True, returns the raw BI distribution object instead of creating a sample
#'   site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Log Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Log Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lognormal}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.log_normal(sample = TRUE)
#' }
#' @export
bi.dist.log_normal=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$log_normal(
       loc = jnp$array(loc),
       scale = jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

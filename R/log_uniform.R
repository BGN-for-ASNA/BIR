#' @title Samples from a Log Uniform distribution.
#'
#' @description
#' The Log Uniform distribution is defined over the positive real numbers and is the result of applying an exponential transformation to a uniform distribution over the interval [low, high]. It is often used when modeling parameters that must be positive.
#'  \deqn{f(x) = \frac{1}{(high - low) \log(high / low)} \text{ for } low \le x \le high}
#'
#' @param low A numeric vector representing the lower bound of the uniform distribution's log-space. Must be positive.
#' @param high A numeric vector representing the upper bound of the uniform distribution's log-space. Must be positive.
#' @param shape A numeric vector specifying the shape of the output. When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Log Uniform distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Log Uniform distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#loguniform}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.log_uniform(1,2, sample = TRUE)
#' }
#' @export
bi.dist.log_uniform=function(low, high, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$log_uniform(
       low = jnp$array(low),
       high = jnp$array(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

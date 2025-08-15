#' @title Relaxed Bernoulli Logits Distribution.
#' @description
#' Represents a relaxed version of the Bernoulli distribution, parameterized by logits and a temperature.
#' The temperature parameter controls the sharpness of the distribution. The distribution is defined
#' by transforming the output of a Logistic distribution through a sigmoid function.
#'
#' \deqn{P(x) = \sigma\left(\frac{x}{\text{temperature}}\right)}
#'
#' @param temperature A numeric vector or matrix representing the temperature parameter, must be positive.
#' @param logits A numeric vector or matrix representing the logits parameter.
#' @param shape A numeric vector specifying the shape of the distribution.  When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
#'   (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector indicating observations to mask.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample
#'   site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Relaxed Bernoulli Logits distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Relaxed Bernoulli Logits distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxed-bernoulli-logits}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.relaxed_bernoulli_logits(1, 0.1, sample = TRUE)
#' }
#' @export
bi.dist.relaxed_bernoulli_logits=function(temperature, logits, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$relaxed_bernoulli_logits(
       temperature = jnp$array(temperature),
       logits = jnp$array(logits),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

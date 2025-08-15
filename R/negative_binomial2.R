#' @title Samples from a Negative Binomial distribution.
#' @description
#' This distribution is parameterized as a Gamma-Poisson with a modified rate.
#' It represents the number of events occurring in a fixed amount of time or trials,
#' where each event has a probability of success.
#'
#'
#' \deqn{P(k) = \frac{\Gamma(k + \alpha)}{\Gamma(k + 1) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^k \left(1 - \frac{\beta}{\alpha + \beta}\right)^k}
#'
#' @param mean A numeric vector, matrix, or array representing the mean of the distribution. This is equivalent to the \eqn{mu} parameter.
#' @param concentration A numeric vector, matrix, or array representing the concentration parameter. This is equivalent to the \eqn{alpha} parameter.
#' @param shape A numeric vector.  Used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional logical vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Negative Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Negative Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#negativebinomial2}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.negative_binomial(mean = 2, concentration = 0, sample = TRUE)
#' }
#' @export
bi.dist.negative_binomial=function(mean, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);

     .bi$dist$negative_binomial2(
       mean = jnp$array(mean),
       concentration = jnp$array(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

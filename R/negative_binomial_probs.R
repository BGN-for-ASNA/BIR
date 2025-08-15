#' @title Sample from a Negative Binomial distribution with probabilities.
#' @description
#' The Negative Binomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is characterized by two parameters: 'concentration' (r) and 'rate' (p).  In this implementation, the 'concentration' parameter is derived from 'total_count' and the 'rate' parameter is derived from 'probs'.
#'
#' \deqn{P(k) = \binom{k + r - 1}{k} p^r (1 - p)^k}
#'
#' @param concentration A numeric vector or array representing the concentration parameter, derived from total_count.
#' @param rate A numeric vector or array representing the rate parameter, derived from probs.
#' @param shape A numeric vector.  When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Negative Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Negative Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#negativebinomialprobs}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.negative_binomial_probs(probs =  c(0.2, 0.3, 0.5), total_count = 10, sample = TRUE)
#' }
#' @export
bi.dist.negative_binomial_probs=function(total_count, probs, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     total_count=jnp$array(as.integer(total_count));

     .bi$dist$negative_binomial_probs(
       total_count,
       probs = jnp$array(probs),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

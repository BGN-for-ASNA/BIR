#' @title Samples from a Negative Binomial Logits distribution.
#'
#' @description
#' The Negative Binomial Logits  distribution is a generalization of the Negative Binomial
#' distribution where the parameter 'r' (number of successes) is expressed as a function
#' of a logit parameter. This allows for more flexible modeling of count data.
#'
#' @param total_count A numeric vector, matrix, or array representing the parameter
#'   controlling the shape of the distribution. Represents the total number of trials.
#' @param logits A numeric vector, matrix, or array representing the log-odds parameter.
#'   Related to the probability of success.
#' @param shape A numeric vector. A multi-purpose argument for shaping. When `sample=False`
#'   (model building), this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX
#'   array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask A logical vector, matrix, or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site. This is essential for building complex distributions
#'   like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Negative Binomial Logits distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Negative Binomial Logits distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso this is a wrapper from \url{https://num.pyro.ai/en/stable/distributions.html#negativebinomiallogits}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.negative_binomial_logits(logits =  c(0.2, 0.3, 0.5), total_count = 10, sample = TRUE)
#' }
#' @export
bi.dist.negative_binomial_logits=function(total_count, logits, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     total_count=jnp$array(as.integer(total_count));

     .bi$dist$negative_binomial_logits(
       total_count,
       logits = jnp$array(logits),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

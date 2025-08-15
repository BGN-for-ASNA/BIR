#' Samples from a MultinomialLogits distribution.
#'
#' This distribution represents the probability of observing a specific outcome from a multinomial experiment,
#' given the logits for each outcome. The logits are the natural logarithm of the odds of each outcome.
#'
#' \deqn{P(k | \mathbf{\pi}) = \frac{n!}{k! (n-k)!} \prod_{i=1}^k \pi_i}
#'
#' @export
#' @importFrom reticulate py_none tuple
#'
#' @param logits A numeric vector, matrix, or array representing the logits for each outcome.
#' @param total_count A numeric vector, matrix, or array representing the total number of trials.
#' @param shape A numeric vector specifying the shape of the distribution.  Use a vector (e.g., `c(10)`) to define the shape.
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'    - When \code{sample=FALSE}, a BI MultinomialLogits distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MultinomialLogits distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#multinomiallogits}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.multinomial_logits(logits =  c(0.2, 0.3, 0.5), total_count = 10, sample = TRUE)
#' }
#' @export
bi.dist.multinomial_logits=function(logits, total_count=1, total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!py$is_none(total_count_max)){total_count_max= as.integer(total_count_max)}
     total_count= jnp$array(as.integer(total_count))

     seed=as.integer(seed);
     .bi$dist$multinomial_logits(
       logits = jnp$array(logits),
       total_count = jnp$array(total_count),
       total_count_max = total_count_max,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

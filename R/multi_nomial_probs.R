#' @title Samples from a Multinomial distribution.
#'
#' @description The Multinomial distribution models the number of times each of several discrete outcomes occurs in a fixed number of trials.  Each trial independently results in one of several outcomes, and each outcome has a probability of occurring.
#'
#' \deqn{P(X = x) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}}
#'
#' where:
#'
#' * \eqn{n} is the total number of trials.
#' * \eqn{x} is a vector of counts for each outcome.
#' * \eqn{p} is a vector of probabilities for each outcome.
#'
#' @param probs A numeric vector of probabilities for each outcome. Must sum to 1.
#' @param total_count The number of trials.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'    - When \code{sample=FALSE}, a BI MultinomialProbs distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MultinomialProbs distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multinomialprobs}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.multinomial_probs(probs =  c(0.2, 0.3, 0.5), total_count = c(10,10), sample = TRUE)
#' }
#' @export
bi.dist.multinomial_probs=function(probs, total_count=1, total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed)
     if(!py$is_none(total_count_max)){total_count_max= as.integer(total_count_max)}
     total_count= jnp$array(as.integer(total_count))

     .bi$dist$multinomial_probs(
       probs = jnp$array(probs),
       total_count= jnp$array(total_count),
       total_count_max= total_count_max,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

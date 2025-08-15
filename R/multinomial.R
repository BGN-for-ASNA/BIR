#' @title Multinomial distribution.
#' @description
#' Samples from a Multinomial distribution, which models the probability of different outcomes in a sequence of independent trials, each with a fixed number of trials and a fixed set of possible outcomes.  It generalizes the binomial distribution to multiple categories.
#' @param total_count An integer or numeric vector representing the number of trials.
#' @param probs A numeric vector representing event probabilities. Must sum to 1.
#' @param logits A numeric vector representing event log probabilities.
#' @param shape A numeric vector used for shaping. When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, optional, to mask observations.
#' @param create_obj A logical value, optional. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Multinomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Multinomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.multinomial(probs = c(0.5,0.1), sample = TRUE)
#' }
#' @export
bi.dist.multinomial=function(total_count=1, probs=py_none(), logits=py_none(), total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     if(!py$is_none(logits)){logits= jnp$array(logits)}
     if(!py$is_none(probs)){probs= jnp$array(probs)}
     .bi$dist$multinomial(total_count=total_count,  probs= probs, logits= logits, total_count_max= total_count_max,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

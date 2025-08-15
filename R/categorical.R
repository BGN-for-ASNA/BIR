#' @title Sample from a Categorical distribution.
#'
#' @description The Categorical distribution, also known as the multinomial distribution,
#' describes the probability of different outcomes from a finite set of possibilities.
#' It is commonly used to model discrete choices or classifications.
#'
#' \deqn{P(k) = \frac{e^{\log(p_k)}}{\sum_{j=1}^{K} e^{\log(p_j)}}}
#'
#' where :math:`p_k` is the probability of outcome :math:`k`, and the sum is over all possible outcomes.
#'
#' @param probs A numeric vector of probabilities for each category. Must sum to 1.
#' @param shape A numeric vector specifying the shape. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array
#'   of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Categorical distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Categorical distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.categorical(probs = c(0.5,0.5), sample = TRUE, shape = c(3))
#' }
#' @export
bi.dist.categorical=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     require(reticulate)
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     py_run_string("def is_none(x): return x is None")
     if(!py$is_none(logits)){logits = jnp$array(logits)}
     if(!py$is_none(probs)){probs = jnp$array(probs)}
     .bi$dist$categorical(probs = probs, logits = logits, validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}




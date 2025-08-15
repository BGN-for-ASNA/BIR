#' @title Samples from a Geometric distribution.
#' @description The Geometric distribution models the number of failures before the first success in a sequence of Bernoulli trials.
#'   It is characterized by a single parameter, the probability of success on each trial.
#'
#' @param probs A numeric vector, matrix, or array representing the probability of success on each trial. Must be between 0 and 1.
#' @param logits A numeric vector, matrix, or array representing the log-odds of success on each trial. `probs = jax.nn.sigmoid(logits)`.
#' @param shape A numeric vector specifying the shape of the output.  Used to set the distribution's batch shape when \code{sample=FALSE} (model building) or as `sample_shape` to draw a raw JAX array of the given shape when \code{sample=TRUE} (direct sampling).
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'    When \code{sample=FALSE}: A BI Geometric distribution object (for model building).
#'
#'    When \code{sample=TRUE}: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
#'
#'    When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#geometric}

#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.geometric(logits = 0.5, sample = TRUE)
#' bi.dist.geometric(probs = 0.5, sample = TRUE)
#' }
#' @export
bi.dist.geometric=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     if(!py$is_none(logits)){logits= jnp$array(logits)}
     if(!py$is_none(probs)){probs= jnp$array(probs)}
     .bi$dist$geometric(probs = probs, logits = logits,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)

}

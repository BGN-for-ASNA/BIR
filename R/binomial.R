#' @title Samples from a Binomial distribution.
#'
#' @description The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials.
#' It represents the probability of obtaining exactly *k* successes in *n* trials, where each trial has a probability *p* of success.
#'
#' \deqn{P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}}
#'
#' @param total_count (int): The number of trials *n*.
#' @param probs (numeric vector, optional): The probability of success *p* for each trial. Must be between 0 and 1.
#' @param logits (numeric vector, optional): The log-odds of success for each trial. `probs = jax.nn.sigmoid(logits)`.
#' @param shape (numeric vector): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (numeric vector of booleans, optional): Optional boolean array to mask observations.
#' @param create_obj (logical, optional): If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @return
#'  - When \code{sample=FALSE}, a BI Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.binomial(probs = jnp$array(c(0.5,0.5)), sample = TRUE)
#' bi.dist.binomial(logits = 1, sample = TRUE)
#' }
#' @export
bi.dist.binomial=function(total_count=1, probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     py_run_string("def is_none(x): return x is None")
     if (py$is_none(logits)){
      .bi$dist$binomial(total_count=jnp$array(as.integer(total_count)),  probs= jnp$array(probs),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
     }else{
       .bi$dist$binomial(total_count=jnp$array(as.integer(total_count)),  logits= jnp$array(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)

     }
}

#' @title Samples from a Binomial distribution.
#'
#' @description The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials.
#' It represents the probability of obtaining exactly *k* successes in *n* trials, where each trial has a probability *p* of success.
#'
#' \deqn{P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}}
#'
#' @param total_count (int): The number of trials *n*.
#' @param probs (numeric vector, optional): The probability of success *p* for each trial. Must be between 0 and 1.
#' @param logits (numeric vector, optional): The log-odds of success for each trial.
#' @param shape (numeric vector): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (numeric vector of booleans, optional): Optional boolean array to mask observations.
#' @param create_obj (logical, optional): If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.binomial(probs = c(0.5,0.5), sample = TRUE)
#' bi.dist.binomial(logits = 1, sample = TRUE)
#' }
#' @export

bi.dist.binomial=function(
    total_count=1L,
    probs=py_none(),
    logits=py_none(),
    validate_args=py_none(),
    name='x',
    obs=py_none(),
    mask=py_none(),
    sample=FALSE,
    seed=py_none(),
    shape=c(),
    event=0,
    create_obj=FALSE
    ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BI_env$.py$is_none(seed)){
       if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     }

     if (.BI_env$.py$is_none(logits)){
       .BI_env$.bi_instance$dist$binomial(
         total_count=.BI_env$jnp$array(total_count),
         probs= .BI_env$jnp$array(probs),
         validate_args= validate_args,
         name= name,  obs= obs,  mask= mask,
         sample= sample,  seed= seed,  shape= shape,
         event= event,  create_obj= create_obj)
     }else{
       .BI_env$.bi_instance$dist$binomial(total_count=.BI_env$jnp$array(total_count),  logits= .BI_env$jnp$array(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)

     }
}



#' @title Samples from a Multinomial distribution.
#'
#' @description
#' The Multinomial distribution models the number of times each of several discrete outcomes occurs in a fixed number of trials.
#' Each trial independently results in one of several outcomes, and each outcome has a probability of occurring.
#'
#' @param probs A numeric vector of probabilities for each outcome. Must sum to 1.
#' @param total_count The number of trials.
#' @param total_count_max (int, optional): An optional integer providing an upper bound on `total_count`. This is used for performance optimization with `lax.scan` when `total_count` is a dynamic JAX tracer, helping to avoid recompilation.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
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
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.multinomial_probs(probs =  c(0.2, 0.3, 0.5), total_count = c(10,10), sample = TRUE)
#' }
#' @export
bi.dist.multinomial_probs=function(probs, total_count=1, total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BI_env$.py$is_none(total_count_max)){total_count_max= as.integer(total_count_max)}
     total_count= .BI_env$jnp$array(as.integer(total_count))

     .BI_env$.bi_instance$dist$multinomial_probs(
       probs = .BI_env$jnp$array(probs),
       total_count= .BI_env$jnp$array(total_count),
       total_count_max= total_count_max,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

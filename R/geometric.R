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
#'    When \code{sample=FALSE}: A BI Geometric distribution object (for model building).
#'
#'    When \code{sample=TRUE}: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
#'
#'    When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#geometric}

#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.geometric(logits = 0.5, sample = TRUE)
#' bi.dist.geometric(probs = 0.5, sample = TRUE)
#' }
#' @export
bi.dist.geometric=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     if(!.BI_env$.py$is_none(logits)){logits= .BI_env$jnp$array(logits)}
     if(!.BI_env$.py$is_none(probs)){probs= .BI_env$jnp$array(probs)}
     .BI_env$.bi_instance$dist$geometric(probs = probs, logits = logits,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)

}

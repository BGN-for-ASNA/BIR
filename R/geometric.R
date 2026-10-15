#' @title Samples from a Geometric distribution.
#' @description
#' Samples from a Geometric distribution, which models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is parameterized by logits, which are transformed into probabilities using the sigmoid function.
#' @param probs A numeric vector, matrix, or array representing the probability of success on each trial. Must be between 0 and 1.
#' @param logits A numeric vector, matrix, or array representing the log-odds of success on each trial. `probs = jax.nn.sigmoid(logits)`.
#' @param shape A numeric vector specifying the shape of the output.  Used to set the distribution's batch shape when \code{sample=FALSE} (model building) or as `sample_shape` to draw a raw JAX array of the given shape when \code{sample=TRUE} (direct sampling).
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'    When \code{sample=FALSE}: A BF Geometric distribution object (for model building).
#'
#'    When \code{sample=TRUE}: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
#'
#'    When \code{create_obj=TRUE}: The raw BF distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#geometric}

#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.geometric(logits = 0.5, sample = TRUE)
#' bf.dist.geometric(probs = 0.5, sample = TRUE)
#' }
#' @export
bf.dist.geometric=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(logits)){logits= .BF_env$jnp$asarray(logits)}
     if(!.BF_env$.py$is_none(probs)){probs= .BF_env$jnp$asarray(probs)}
     .BF_env$.bf_instance$dist$geometric(probs = probs, logits = logits,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)

}

#' @title Samples from a Relaxed Bernoulli distribution.
#' @description
#' The Relaxed Bernoulli is a continuous distribution on the interval \deqn{(0,1)}
#' that smoothly approximates the discrete Bernoulli distribution (which has support \deqn{{0,1}}).
#' It was introduced to allow for *differentiable* sampling of approximate binary random variables,
#' which is useful in variational inference and other gradient-based optimization settings.
#'
#' @param temperature A numeric value representing the temperature parameter.
#' @param probs (jnp.ndarray, optional): The probability of success. Must be in the interval `[0, 1]`. Only one of `probs` or `logits` can be specified.
#' @param logits A numeric vector or matrix representing the logits parameter.
#' @param shape A numeric vector (e.g., `c(10)`) specifying the shape. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BF Relaxed Bernoulli distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Relaxed Bernoulli distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli}
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.relaxed_bernoulli(temperature = c(1,1), logits = 0.0, sample = TRUE)
#' }
#' @export
bf.dist.relaxed_bernoulli=function(temperature, probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(logits)){logits=.BF_env$jnp$asarray(logits)}
     if(!.BF_env$.py$is_none(probs)){probs=.BF_env$jnp$asarray(probs)}
     .BF_env$.bf_instance$dist$relaxed_bernoulli(
       temperature = .BF_env$jnp$asarray(temperature),
       probs = probs,
       logits = logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

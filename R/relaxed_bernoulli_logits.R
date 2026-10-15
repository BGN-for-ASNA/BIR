#' @title Relaxed Bernoulli Logits Distribution.
#' @description
#' The Relaxed Bernoulli (logits) is a **continuous** relaxation of the standard Bernoulli distribution,
#' parameterised by logits (or probabilities) and a **temperature** parameter.
#' Rather than producing strictly 0 or 1, it produces values in the continuous interval (0, 1).
#' As the temperature \eqn{\to 0}, the distribution approximates a Bernoulli; as temperature \eqn{\to \infty}, the distribution approximates a uniform distribution.
#'
#' @param temperature A numeric vector or matrix representing the temperature parameter, must be positive.
#' @param logits A numeric vector or matrix representing the logits parameter.
#' @param shape A numeric vector specifying the shape of the distribution.  When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
#'   (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector indicating observations to mask.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample
#'   site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BF Relaxed Bernoulli Logits distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Relaxed Bernoulli Logits distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxed-bernoulli-logits}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.relaxed_bernoulli_logits(1, 0.1, sample = TRUE)
#' }
#' @export
bf.dist.relaxed_bernoulli_logits=function(temperature, logits, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$relaxed_bernoulli_logits(
       temperature = .BF_env$jnp$asarray(temperature),
       logits = .BF_env$jnp$asarray(logits),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

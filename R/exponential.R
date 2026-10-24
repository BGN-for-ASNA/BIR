#' @title Exponential distribution
#'
#' @description
#' The Exponential distribution is a continuous probability distribution that
#' models the time until an event occurs in a Poisson process, where events
#' occur continuously and independently at a constant average rate. It is often
#' used to model the duration of events, such as the time until a machine fails
#' or the length of a phone call.
#'
#' For ( x \\ge 0 )
#'
#' \deqn{F(x; \lambda) = 1 - e^{-\lambda x}}
#'
#' Zhere ( \\lambda > 0 ) is the rate parameter (the mean is ( 1/\\lambda )).
#'
#' @param rate The rate parameter, $\\lambda`. Must be positive.
#' @param validate_args Logical. Whether to validate the parameter values.
#'   Defaults to `py_none()` (the NumPyro default).
#' @param name Character. Name of the random variable in the model. When not
#'   given, an observed site is named after the variable passed to `obs` (`obs =
#'   y` gives `"y"`); use [bf_model()] to name parameters after the variables they
#'   are assigned to.
#' @param obs Numeric vector or array of observed values. If given, the random
#'   variable is conditioned on them (likelihood); otherwise it is latent.
#'   Defaults to `py_none()`.
#' @param mask Optional logical vector or array masking observations (`FALSE` =
#'   ignored). Defaults to `py_none()`.
#' @param sample Logical. If `TRUE`, directly draw values from the distribution
#'   and return them; if `FALSE` (default), create a random variable inside a
#'   model.
#' @param seed Integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param shape Integer vector. With `sample = FALSE`, the batch shape of the
#'   distribution in the model; with `sample = TRUE`, the shape of the draw.
#'   Defaults to `c()` (no extra dimensions).
#' @param event Integer. Number of batch dimensions reinterpreted as event
#'   dimensions (model building only). Defaults to `0`.
#' @param create_obj Logical. If `TRUE`, return the distribution object itself
#'   instead of a random variable (e.g. to build mixtures). Defaults to `FALSE`.
#' @param to_jax Logical. Return a JAX array (`TRUE`, default) or convert to R.
#'
#' @return
#' \itemize{
#'   \item When `sample = FALSE`: A BF Exponential distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Exponential
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.exponential(rate = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#exponential}
#'
#' @references
#' https://reference.wolfram.com/language/ref/ExponentialDistribution.html
#'
#' https://en.wikipedia.org/wiki/Exponential_distribution
#'
#' @export
bf.dist.exponential=function(rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$exponential(
       rate=.BF_env$jnp$asarray(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

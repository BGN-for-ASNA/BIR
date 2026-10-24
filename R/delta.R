#' @title Delta distribution
#'
#' @description
#' The Delta distribution, also known as a point mass distribution, assigns
#' probability 1 to a single point and 0 elsewhere. It's useful for representing
#' deterministic variables or as a building block for more complex
#' distributions.
#'
#' \deqn{F(x) = \begin{cases} 0, & x < a, \ 1, & x \ge a. \end{cases}}
#'
#' @param v The location of the point mass.
#' @param log_density The log probability density of the point mass. This is
#'   primarily for creating distributions that are non-normalized or for specific
#'   advanced use cases. For a standard delta distribution, this should be 0.
#'   Defaults to 0.0.
#' @param event_dim The number of rightmost dimensions of `v` to interpret as
#'   event dimensions. Defaults to 0.
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
#'   \item When `sample = FALSE`: A BF Circulant Delta Distribution distribution
#'     object (for  model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     BernoulliLogits distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF Circulant Delta Distribution
#'     object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.delta(v = 0.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#delta}
#'
#' @export
bf.dist.delta=function(v=0.0, log_density=0.0, event_dim=0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event_dim=as.integer(event)
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$delta(
       v=.BF_env$jnp$asarray(v),
       log_density= .BF_env$jnp$asarray(log_density),
       event_dim= .BF_env$jnp$asarray(event_dim),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

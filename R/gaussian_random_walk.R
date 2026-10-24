#' @title Gaussian Random Walk distribution
#'
#' @description
#' Creates a distribution over a Gaussian random walk of a specified number of
#' steps. This is a discrete-time stochastic process where the value at each
#' step is the previous value plus a Gaussian-distributed increment. The
#' distribution is over the entire path.
#'
#' @param scale The standard deviation ($\\sigma`) of the Gaussian increments.
#'   Must be positive. Defaults to 1.0. - *num_steps (int, optional): The number
#'   of steps in the random walk, which determines the event shape of the
#'   distribution. Must be positive. Defaults to 1. - *validate_args (bool,
#'   optional): Whether to enable validation of distribution parameters. Defaults
#'   to `py_none()`.
#' @param num_steps See the Python documentation of `num_steps`.
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
#'   \item BF.primitives.Messenger: A BF sample site object when used in a model
#'     context (`sample = FALSE`).
#'   \item numeric vector or array: A JAX array of samples drawn from the
#'     GaussianRandomWalk distribution (for direct sampling, `sample = TRUE`).
#'   \item numpyro.distributions.Distribution: The raw BF distribution object (if
#'     `create_obj = TRUE`).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' path <- bf.dist.gaussian_random_walk(scale = 0.5, num_steps = 100L, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#gaussianrandomwalk}
#'
#' @export
bf.dist.gaussian_random_walk=function(scale=1.0, num_steps=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  #return(print("No more available since jax > 0.06"))
  shape=do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
  num_steps=as.integer(num_steps);
  .BF_env$.bf_instance$dist$gaussian_random_walk(
       scale = .BF_env$jnp$asarray(scale),
       num_steps = num_steps,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

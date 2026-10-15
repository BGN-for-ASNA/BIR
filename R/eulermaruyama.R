#' @title Euler–Maruyama distribution
#'
#' @description
#' Euler–Maruyama methode is a method for the approximate numerical solution of
#' a stochastic differential equation (SDE). It simulates the solution to an SDE
#' by iteratively applying the Euler method to each time step, incorporating a
#' random perturbation to account for the diffusion term.
#'
#' \deqn{dX_t = f(X_t, t) dt + g(X_t, t) dW_t}
#'
#' where:
#' - \eqn{X_t` is the state of the system at time}t`.
#' - $f(X_t, t)` is the drift coefficient.
#' - $g(X_t, t)` is the diffusion coefficient.
#' - $dW_t` is a Wiener process (Brownian motion).
#'
#' @param t Discretized time steps.
#' @param sde_fn A function that takes the current state and time as input and
#'   returns the drift and diffusion coefficients.
#' @param init_dist The initial distribution of the system.
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
#'   \item numeric vector or array: Samples drawn from the Euler–Maruyama
#'     distribution.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.euler_maruyama(t = jnp$array(c(0.0, 0.1, 0.2)), sde_fn = function(x, t) reticulate::tuple(x, 1.0), init_dist = bf.dist.normal(0.0, 1.0, create_obj = TRUE), sample = TRUE)
#' }
#'
#' @export
bf.dist.euler_maruyama=function(t, sde_fn, init_dist, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(reticulate::tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     .BF_env$.bf_instance$dist$euler_maruyama(
       t = .BF_env$jnp$asarray(t),
       sde_fn = sde_fn,
       init_dist = init_dist,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

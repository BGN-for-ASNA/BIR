#' @title Logistic distribution
#'
#' @description
#' Samples from a Logistic distribution.
#'
#' The Logistic distribution is a continuous probability distribution defined by
#' two parameters: location and scale. It is often used to model growth
#' processes and is closely related to the normal distribution.Its CDF is the
#' logistic (sigmoid) function, which makes it appealing in modeling
#' probabilities, logistic regression, and various growth models. It resembles
#' the normal distribution in shape (bell‐shaped, symmetric) but has **heavier
#' tails** (i.e. more probability in the extremes) and simpler closed‐form
#' expressions for the CDF.
#'
#' * **Location parameter**: \eqn{\mu \in \mathbb{R}}\ — the center or “mean” of
#' the distribution.
#' * **Scale parameter**: \eqn{\s > 0}\ — controls the spread (similar role to
#' standard deviation).
#' * **Support**: \eqn{\ x \in (-\infty, +\infty)}\.
#'
#' Let (X \\sim \\mathrm\{Logistic\}(\\mu, s)). Then:
#'
#' \deqn{F(x; \mu, s) = \frac{1}{1 + \exp\bigl(-\frac{x - \mu}{s}\bigr)}.}
#'
#' Equivalently:
#'
#' \deqn{F(x) = \frac12 + \frac12 \tanh!\bigl(\frac{x - \mu}{2s}\bigr).}
#'
#' @param loc The location parameter, specifying the median of the distribution.
#'   Defaults to 0.0.
#' @param scale The scale parameter, which determines the spread of the
#'   distribution. Must be positive. Defaults to 1.0.
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
#'   \item BF Logistic distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the Logistic distribution (for direct
#'     sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.logistic(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#logistic}
#'
#' @references
#' https://en.wikipedia.org/wiki/Logistic_distribution
#'
#' @export
bf.dist.logistic <- function(loc = 0.0, scale = 1.0, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$logistic(
    loc = .BF_env$jnp$asarray(loc),
    scale = .BF_env$jnp$asarray(scale),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

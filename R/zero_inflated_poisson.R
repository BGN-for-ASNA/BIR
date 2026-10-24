#' @title A Zero Inflated Poisson distribution
#'
#' @description
#' The Zero-Inflated Poisson distribution is a discrete count‐distribution
#' designed for data with *more zeros* than would be expected under a standard
#' Poisson. Essentially, it assumes two underlying processes:
#'
#' * With probability \eqn{\pi} you are in a “structural zero” state (i.e., you
#' automatically get a zero count).
#' * With probability \eqn{1 - \pi} you draw from a standard Poisson
#' distribution with parameter \eqn{\lambda}.
#'
#' This results in a mixture distribution that places more mass at zero than a
#' Poisson alone would. It’s widely used in, for instance, ecology (species
#' counts with many zeros), insurance/claims problems, and any count‐data
#' setting with excess zeros.
#'
#' The probability density function (PDF) is given by:
#'
#' Let \eqn{X} be a random variable following a Zero-Inflated Poisson with
#' parameters
#'
#' * \eqn{\pi\in[0,1]}: zero-inflation probability (chance of being structural
#' zero)
#' * \eqn{\lambda>0}: mean of the Poisson component
#'
#' Then:
#'
#' \deqn{P(X = k) = \begin{cases} \pi ;+; (1 - \pi),e^{-\lambda}, & \text{if } k = 0, [6pt] (1 - \pi),\dfrac{\lambda^k e^{-\lambda}}{k!}, & \text{if } k = 1,2,3,\dots \end{cases}}
#'
#' (Here the Poisson component contributes also a mass at zero of \eqn{(1 -
#' \pi)e^{-\lambda}}, plus the extra \eqn{\pi} mass.)
#'
#' @param gate The probability of observing a zero.
#' @param rate The rate parameter of the underlying Poisson distribution.
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
#'   \item BF ZeroInflatedPoisson distribution object (when `sample = FALSE`).
#'   \item JAX array of samples drawn from the ZeroInflatedPoisson distribution
#'     (when `sample = TRUE`).
#'   \item The raw BF distribution object (when `create_obj = TRUE`).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.zero_inflated_poisson(gate = 0.3, rate = 2.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflatedpoisson}
#'
#' @references
#' https://en.wikipedia.org/wiki/Zero-inflated_model "Zero-inflated model"
#'
#' https://www.statisticshowto.com/zero-inflated-poisson-distribution
#' "Zero-Inflated Poisson distribution - Statistics How To"
#'
#' @export
bf.dist.zero_inflated_poisson <- function(gate, rate = 1.0, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$zero_inflated_poisson(
    gate = .BF_env$jnp$asarray(gate),
    rate = .BF_env$jnp$asarray(rate),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj
  )
}

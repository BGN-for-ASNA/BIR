#' @title Levy distribution
#'
#' @description
#' Samples from a Levy distribution. The Lévy distribution is a continuous
#' probability distribution on the positive real line (or shifted positive line)
#' that is *heavy-tailed*, *skewed*, and arises naturally in connection with
#' stable distributions specifically the case with stability index \eqn{\alpha =
#' \tfrac12}. It is often used in contexts such as hitting‐time problems for
#' Brownian motion, physics (e.g., van der Waals line‐shapes), and modelling
#' very heavy‐tailed phenomena. Let (X) be a Lévy‐distributed random variable
#' with **location** parameter \eqn{\mu} and **scale** parameter (c > 0). The
#' support is \eqn{x \ge \mu}.
#'
#' **PDF**
#'
#' \deqn{f(x; \mu, c) = \sqrt{\frac{c}{2\pi}} ; \frac{1}{(x - \mu)^{3/2}} ; \exp!\Bigl( -\frac{c}{2 (x - \mu)} \Bigr), \quad x \ge \mu.}
#'
#' **CDF**
#'
#' \deqn{F(x; \mu, c) = \Pr(X \le x) = \operatorname{erfc}!\Bigl(\sqrt{\tfrac{c}{2 (x - \mu)}}\Bigr) = 2 - 2,\Phi!\Bigl(\sqrt{\tfrac{c}{(x - \mu)}}\Bigr), \quad x \ge \mu.}
#'
#' Here \eqn{\operatorname{erfc}} is the complementary error function and
#' \eqn{\Phi} is the standard normal CDF.
#'
#' In the *standard* case with \eqn{\mu = 0} and \eqn{c = 1}, one often simply
#' writes \eqn{X \sim \text{Lévy}(0,1)}.
#'
#' @param loc Location parameter.
#' @param scale See the Python documentation of `scale`.
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
#'   \item BF Levy distribution object: When `sample = FALSE` (for model
#'     building).
#'   \item JAX array: When `sample = TRUE` (for direct sampling).
#'   \item BF distribution object: When `create_obj = TRUE` (for advanced use
#'     cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.levy(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#levy}
#'
#' @export
bf.dist.levy <- function(loc, scale, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$levy(
    loc = .BF_env$jnp$asarray(loc),
    scale = .BF_env$jnp$asarray(scale),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

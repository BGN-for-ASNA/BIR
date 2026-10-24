#' @title Lower Truncated Power Law distribution
#'
#' @description
#' The *Lower-Truncated Power-Law* distribution (also known as the *Pareto Type
#' I* or *power-law with a lower bound*) models quantities that follow a
#' heavy-tailed power-law behavior but are bounded below by a minimum value (
#' x_\{\\min\} ). It is commonly used to describe phenomena such as wealth
#' distributions, city sizes, and biological scaling laws.
#'
#' The probability density function (PDF) is given by:
#'
#' For ( x \\ge x_\{\\min\} > 0 ):
#'
#' \deqn{f(x \mid \alpha, x_{\min}) = \frac{\alpha - 1}{x_{\min}} \left( \frac{x_{\min}}{x} \right)^{\alpha}}
#'
#' where \eqn{\alpha > 1} is the shape (scaling) parameter.
#'
#' @param alpha See the Python documentation of `alpha`.
#' @param low See the Python documentation of `low`.
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
#'   \item When `sample = FALSE`: A BF LowerTruncatedPowerLaw distribution object
#'     (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     LowerTruncatedPowerLaw distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.lower_truncated_power_law(alpha = -2.0, low = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#lowertruncatedpowerlaw}
#'
#' @references
#' [Power-law distribution –
#' Wikipedia](https://en.wikipedia.org/wiki/Power_law#Continuous_power-law_distribution)
#'
#' [Pareto distribution –
#' Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution)
#'
#' [Truncated Pareto distribution](https://arxiv.org/abs/0706.1062)
#'
#' @export
bf.dist.lower_truncated_power_law <- function(alpha, low, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$lower_truncated_power_law(
    alpha = .BF_env$jnp$asarray(alpha),
    low = .bf_arr(low),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

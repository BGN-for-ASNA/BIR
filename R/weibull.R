#' @title Weibull distribution
#'
#' @description
#' Samples from a Weibull distribution.
#'
#' The Weibull distribution is widely used for modeling **lifetime or
#' reliability data**. Its shape parameter (k) controls the hazard function:
#'
#' * (k < 1): decreasing hazard (infant mortality)
#' * (k = 1): constant hazard → reduces to **Exponential distribution**
#' * (k > 1): increasing hazard (aging/failure over time)
#'
#' \deqn{f(x \mid \lambda, k) = \begin{cases} \dfrac{k}{\lambda} \left(\dfrac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, & x \ge 0, [2mm] 0, & x < 0 \end{cases}}
#'
#' @param scale The scale parameter of the Weibull distribution. Must be
#'   positive.
#' @param concentration The concentration parameter of the Weibull distribution.
#'   Must be positive.
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
#'   \item BF Weibull distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the Weibull distribution (for direct
#'     sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.weibull(scale = 1.0, concentration = 2.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#weibull}
#'
#' @references
#' [Wikipedia – Weibull
#' distribution](https://en.wikipedia.org/wiki/Weibull_distribution#Probability_density_function)
#'
#' Weibull, W. (1951). *A Statistical Distribution Function of Wide
#' Applicability.* Journal of Applied Mechanics.
#'
#' @export
bf.dist.weibull=function(scale, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$weibull(
       scale = .BF_env$jnp$asarray(scale),
       concentration = .BF_env$jnp$asarray(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

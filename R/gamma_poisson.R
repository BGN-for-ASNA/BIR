#' @title Gamma Poisson distribution
#'
#' @description
#' The Gamma-Poisson distribution, also known as the Negative Binomial
#' distribution, models overdispersed count data. It arises from a hierarchical
#' process where the rate parameter of a Poisson distribution is itself a random
#' variable following a Gamma distribution. This structure allows the model to
#' capture variability in count data that exceeds what is predicted by the
#' Poisson distribution, making it suitable for applications like modeling
#' RNA-sequencing data and microbial count.
#'
#' \deqn{f(x) = \frac{\Gamma(x + \beta) \alpha^x}{\Gamma(\beta)(1 + \alpha)^{\beta + x} x!}, \quad x = 0, 1, 2, \ldots}
#'
#' Where \eqn{\Gamma} denotes the gamma function.
#'
#' This CDF expresses the probability that the count variable ( X ) takes a
#' value less than or equal to ( x ), incorporating the overdispersion
#' introduced by the Gamma distribution.
#'
#' @param concentration Shape parameter (alpha) of the Gamma distribution.
#' @param rate Rate parameter (beta) for the Gamma distribution.
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
#'   \item When `sample = FALSE`: A BF GammaPoisson distribution object (for model
#'   \item building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     GammaPoisson
#'   \item distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'   \item use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.gamma_poisson(concentration = 1.0, rate = 2.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#gammapoisson}
#'
#' @references
#' \[1\] https://www.statisticshowto.com/gamma-poisson-distribution
#'
#' https://en.wikipedia.org/wiki/Negative_binomial_distribution
#'
#' @export
bf.dist.gamma_poisson=function(concentration, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$gamma_poisson(
       concentration = .BF_env$jnp$asarray(concentration),
       rate= .BF_env$jnp$asarray(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

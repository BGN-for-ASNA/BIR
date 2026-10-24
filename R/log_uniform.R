#' @title Log-Uniform distribution
#'
#' @description
#' Samples from a Log Uniform distribution.
#'
#' The Log Uniform distribution is defined over the positive real numbers and is
#' the result of applying an exponential transformation to a uniform
#' distribution over the interval \[low, high\]. It is often used when modeling
#' parameters that must be positive.
#'
#' A random variable \eqn{X} is **log-uniform** on \eqn{[a, b]}, with \eqn{0 < a
#' < b}, if \eqn{\ln X} is uniformly distributed on \eqn{[\ln a, \ln b]}.
#' Equivalently, the density of \eqn{X} is proportional to \eqn{1/x} over that
#' interval. This distribution is sometimes called the *reciprocal
#' distribution*. It is useful in modeling scales spanning several orders of
#' magnitude, where you want every decade (or log-interval) to have equal
#' weight.
#'
#' For \eqn{x \in [a, b]}:
#'
#' * **PDF**
#'
#' \deqn{f(x) = \frac{1}{x , [\ln(b) - \ln(a)]}, \quad a \le x \le b.}
#'
#' Outside \eqn{[a, b]}, \eqn{f(x) = 0}.
#'
#' * **CDF**
#'
#' \deqn{F(x) = \Pr(X \le x) = \frac{\ln(x) - \ln(a)}{\ln(b) - \ln(a)}, \quad a \le x \le b.}
#'
#' (For \eqn{x < a}, \eqn{F(x) = 0}; for \eqn{x > b}, \eqn{F(x) = 1}.
#'
#' @param low The lower bound of the uniform distribution's log-space. Must be
#'   positive.
#' @param high The upper bound of the uniform distribution's log-space. Must be
#'   positive.
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
#'   \item BF LogUniform distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the LogUniform distribution (for direct
#'     sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.log_uniform(low = 0.1, high = 10.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#loguniform}
#'
#' @references
#' https://en.wikipedia.org/wiki/Reciprocal_distribution
#'
#' https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html
#'
#' https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_loguniform.html
#'
#' @export
bf.dist.log_uniform=function(low, high, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$log_uniform(
       low = .bf_arr(low),
       high = .bf_arr(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

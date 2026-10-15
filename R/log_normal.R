#' @title Log-Normal distribution
#'
#' @description
#' The Log-Normal distribution is a probability distribution defined for
#' positive real-valued random variables, parameterized by a location parameter
#' (loc : \eqn{\mu}) and a scale parameter (scale).  It arises when the
#' logarithm of a random variable is normally distributed.
#' * A random variable (X) is **log-normal** if its natural logarithm (\\ln X)
#' is (approximately) normally distributed.  Equivalently, one can write:
#'
#' \deqn{X = \exp(Y), \quad \text{where } Y \sim \mathcal{N}(\mu, \sigma^2).}
#'
#' Because (X) is the exponential of a normal, it is always positive (support
#' (x>0)). It is useful in contexts where growth, multiplicative effects, or
#' compounding factors dominate (e.g. stock prices, income, sizes of biological
#' organisms).
#'
#' @param loc Location parameter.
#' @param scale Scale parameter.
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
#'   \item BF LogNormal distribution object (for model building).
#'   \item JAX array of samples drawn from the LogNormal distribution (for direct
#'     sampling).
#'   \item The raw BF distribution object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.log_normal(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#lognormal}
#'
#' @references
#' https://en.wikipedia.org/wiki/Log-normal_distribution
#'
#' @export
bf.dist.log_normal=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$log_normal(
       loc = .BF_env$jnp$asarray(loc),
       scale = .BF_env$jnp$asarray(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

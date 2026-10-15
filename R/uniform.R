#' @title Uniform distribution
#'
#' @description
#' The Uniform distribution is the simplest continuous distribution: every value
#' in the interval (\[a, b\]) is **equally likely**. It is widely used for
#' modeling complete randomness within a fixed range, random sampling, and as a
#' building block for other distributions.
#'
#' \deqn{f(x) = \begin{cases} \dfrac{1}{b - a}, & a \le x \le b, [2mm] 0, & \text{otherwise}. \end{cases}}
#'
#' @param low The lower bound of the uniform interval.
#' @param high The upper bound of the uniform interval.
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
#'   \item BF Uniform distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the Uniform distribution (for direct
#'   \item sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'   \item `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.uniform(low = 0.0, high = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#uniform}
#'
#' @references
#' [Wikipedia – Uniform distribution
#' (continuous)](https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29)
#'
#' @export
bf.dist.uniform=function(low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$uniform(
       low = .bf_arr(low),
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

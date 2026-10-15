#' @title Two Sided Truncated distribution
#'
#' @description
#' A “two-sided truncated distribution” is a general concept: you take a base
#' continuous distribution and **restrict it** to an interval (\[`low`,
#' `high`\]), discarding all mass outside, then **renormalize** so the inner
#' portion integrates to 1. I’ll spell out the general formulas, caveats,
#' sampling strategies, and special cases (e.g. truncated normal) to illustrate.
#'
#' \deqn{f(x) = \begin{cases} \dfrac{p(x)}{P(\text{low} \le X \le \text{high})}, & \text{if } \text{low} \le x \le \text{high}, \\[6pt] 0, & \text{otherwise}. \end{cases}}
#'
#' where \eqn{p(x)} is the probability density function of the base
#' distribution.
#'
#' @param base_dist The base distribution to truncate.
#' @param low The lower bound for truncation.
#' @param high The upper bound for truncation.
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
#'   \item When `sample = FALSE`: A BF TwoSidedTruncatedDistribution distribution
#'     object (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     TwoSidedTruncatedDistribution distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.two_sided_truncated_distribution(base_dist = bf.dist.normal(0.0, 1.0, create_obj = TRUE), low = -1.0, high = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution}
#'
#' @export
bf.dist.two_sided_truncated_distribution=function(base_dist, low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$two_sided_truncated_distribution(
       base_dist,
       low= .bf_arr(low),
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

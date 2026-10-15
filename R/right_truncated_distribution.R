#' @title Right Truncated distribution
#'
#' @description
#' A right-truncated distribution is a statistical distribution that arises when
#' the possible values  of a random variable are restricted to be below a
#' certain specified value `high`. In essence, the right tail of the original
#' distribution is "cut off" at a particular point, and the remaining
#' probability is redistributed among the allowable values. This type of
#' distribution is common in various fields where there are inherent upper
#' limits or observational constraints.
#'
#' The probability density function (PDF) for a continuous right-truncated
#' distribution or the probability mass function (PMF) for a discrete one is
#' derived from the original distribution by normalizing it over the restricted
#' range.
#'
#' **For a continuous random variable X:**
#'
#' If the original probability density function is `f(x)` and the cumulative
#' distribution function is `F(x)`, and the distribution is right-truncated at a
#' value `b` (meaning `x ≤ b`), the new PDF, `f_T(x)`, is:
#'
#' \deqn{f_{\rm trunc}(x \mid X \le b) = \begin{cases} \displaystyle \frac{,f(x),}{,F(b),}, & x \le b, \ 0, & x > b. \end{cases}}
#'
#' **For a discrete random variable X:**
#'
#' If the original probability mass function is `P(X = k)` and the cumulative
#' distribution function is `F(k)`, and the distribution is right-truncated at a
#' value `b` (meaning `k ≤ b`), the new PMF, `P_T(X = k)`, is:
#'
#' \deqn{P_T(X=k) = \begin{cases} \frac{P(X=k)}{F(b)} & \text{if } k \le b \\ 0 & \text{if } k > b \end{cases}}
#'
#' @param base_dist The base distribution to truncate.  Must be a univariate
#'   distribution with real support.
#' @param high The upper truncation point. The support of the new distribution
#'   is \eqn{(-\infty, \text{high}]}. Defaults to 0.0.
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
#'   \item When `sample = FALSE`: A BF RightTruncatedDistribution distribution
#'     object (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     RightTruncatedDistribution distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.right_truncated_distribution(base_dist = bf.dist.normal(0, 1, create_obj = TRUE), high = 0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#righttruncateddistribution}
#'
#' @references
#' https://en.wikipedia.org/wiki/Truncated_normal_distribution
#'
#' @export
bf.dist.right_truncated_distribution=function(base_dist, high=0.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$right_truncated_distribution(
       base_dist,
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

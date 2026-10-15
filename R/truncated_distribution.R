#' @title Truncated distribution
#'
#' @description
#' Samples from a Truncated Distribution.
#'
#' A **truncated distribution** arises when you take a random variable \eqn{X}
#' that originally has some distribution (with PDF \eqn{f_X(x)} and CDF
#' \eqn{F_X(x)}) and you restrict attention only to those values of \eqn{X} that
#' are *above* a given truncation point \eqn{a}. In other words you only observe
#' \eqn{X} when \eqn{X > a}. All the “mass” below (or equal to) \eqn{a} is
#' **excluded** (not just unobserved, but removed from the sample/analysis).
#' This differs from *censoring*, where values below a threshold might be known
#' (for example “< a”), but here they are entirely excluded from the domain.
#' Left truncation is common in many applied fields — for instance:
#'
#' * In survival analysis: subjects whose event time happens before the study
#' start are not included.
#' * In insurance or losses: only losses above a deductible (threshold) are
#' recorded, so the loss distribution is left-truncated at that deductible.
#' * In industrial/life-data: items used before data collection start, so only
#' those with lifetime > some lower bound are observed.
#'
#' Let (X) be a random variable with PDF. Choose truncation points \eqn{a < X
#' \le b}. Then  the truncated distribution (Y = X \\mid (a < X \\le b)) has:
#'
#' \deqn{f_Y(y) = \frac{f_X(y)}{F_X(b) - F_X(a)}, \quad \text{for } a < y \le b,}
#'
#' and (0) outside that interval.
#' * The lower bound (a) can be (-\\infty) (so only an upper truncation), and
#' the upper bound (b) can be (+\\infty) (so only a lower truncation).
#' (\[Statistics How To\]\[2\])
#' * If both bounds are finite (i.e., doubly truncated), that is a valid and
#' often‐used scenario.
#'
#' @param base_dist The base distribution to be truncated. This should be a
#'   univariate distribution. Currently, only the following distributions are
#'   supported: Cauchy, Laplace, Logistic, Normal, and StudentT.
#' @param low See the Python documentation of `low`.
#' @param high See the Python documentation of `high`.
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
#'   \item When `sample = FALSE`: A BF TruncatedDistribution distribution object
#'     (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     TruncatedDistribution distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.truncated_distribution(base_dist = bf.dist.normal(0, 1, create_obj = TRUE), high = 1, low = 0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#truncateddistribution}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Truncated_distribution "Truncated
#' distribution"
#'
#' \[2\]:
#' https://pages.stern.nyu.edu/~wgreene/Text/Edition8/PDF/M19_GREE1366_08_SE_C19.pdf
#' "19"
#'
#' \[3\]:
#' https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/dist.Truncated.html
#' "R: Truncated Distributions"
#'
#' \[4\]:
#' https://dspace.cuni.cz/bitstream/handle/20.500.11956/127921/130308341.pdf?isAllowed=y&sequence=1&utm_source.com
#' "Truncated data thesis"
#'
#' @export
bf.dist.truncated_distribution=function(base_dist, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$truncated_distribution(
       base_dist,
       low = .bf_arr(low),
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Truncated Normal distribution
#'
#' @description
#' A truncated normal distribution is derived from a normal (Gaussian) random
#' variable by restricting (truncating) its domain to an interval \eqn{[a, b]}
#' (which could be one‐sided, e.g., (a) only or (b) only). It is defined by its
#' location (`loc`), scale (`scale`), lower bound  \eqn{a} (`low`), and upper
#' bound \eqn{b} (`high`). In effect: if \eqn{X \sim \mathcal N(\mu, \sigma^2)},
#' then the truncated version \eqn{Y = X | (a \le X \le b)} has the same “shape”
#' but only supports values in \eqn{[a,b]}. This is used when you know that
#' values outside a range are impossible or not observed (e.g., measurement
#' limits, natural bounds).
#'
#' Let \eqn{X \sim \mathcal N(\mu, \sigma^2)}. Define truncation bounds \eqn{a <
#' b} (could be \eqn{a = -\infty} or \eqn{b = +\infty} for one‐sided
#' truncation). Then for \eqn{y \in [a,b]},
#'
#' \deqn{f_Y(y) = \frac{1}{\sigma} ; \frac{\varphi!\bigl(\frac{y - \mu}{\sigma}\bigr)}{\Phi!\bigl(\frac{b - \mu}{\sigma}\bigr) - \Phi!\bigl(\frac{a - \mu}{\sigma}\bigr)} ,}
#'
#' where:
#'
#' * \eqn{\varphi(z)} is the standard normal PDF, \eqn{\varphi(z) =
#' \frac1{\sqrt{2\pi}} e^{-z^2/2}}.
#' * \eqn{\Phi(z)} is the standard normal CDF.
#' * The denominator \eqn{\Phi\bigl(\tfrac{b - \mu}{\sigma}\bigr) -
#' \Phi\bigl(\tfrac{a - \mu}{\sigma}\bigr)} normalises the density so the total
#' probability over (\[a,b\]) is 1.
#'
#' For values (y < a) or (y > b), (f_Y(y) = 0).
#'
#' @param loc The location parameter of the normal distribution.
#' @param scale See the Python documentation of `scale`.
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
#'   \item BF TruncatedNormal distribution object (for model building).
#'   \item JAX array of samples drawn from the TruncatedNormal distribution (for
#'   \item direct sampling).
#'   \item The raw BF distribution object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.truncated_normal(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#truncatednormal_lowercase}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Truncated_normal_distribution "Truncated
#' normal distribution"
#'
#' \[2\]:
#' https://people.math.sc.edu/burkardt/m_src/truncated_normal/truncated_normal.html
#' "TRUNCATED_NORMAL - The Truncated Normal Distribution"
#'
#' \[3\]: https://www.statisticshowto.com/truncated-normal-distribution/
#' "Truncated Distribution / Truncated Normal Distribution - Statistics ..."
#'
#' \[4\]:
#' https://real-statistics.com/normal-distribution/truncated-normal-distribution/
#' "Truncated Normal Distribution | Real Statistics Using Excel"
#'
#' \[5\]:
#' https://stats.stackexchange.com/questions/525894/understanding-the-pdf-of-a-truncated-normal-distribution
#' "Understanding the pdf of a truncated normal distribution"
#'
#' @export
bf.dist.truncated_normal=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$truncated_normal(
       loc=.BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       low= .bf_arr(low),
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

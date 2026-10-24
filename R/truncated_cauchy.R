#' @title Truncated Cauchy distribution
#'
#' @description
#' The Cauchy distribution, also known as the Lorentz distribution, is a
#' continuous probability distribution that appears frequently in various areas
#' of mathematics and physics. It is characterized by its heavy tails, which
#' extend to infinity. The truncated version limits the support of the Cauchy
#' distribution to a specified interval.
#'
#' Let the base Cauchy have parameters \eqn{\mu} (location) and \eqn{\gamma > 0}
#' (scale). Let the truncation bounds be \eqn{a} (`low`) and \eqn{b} (`high`)
#' (with \eqn{a < b}). Then the PDF of the truncated Cauchy at \eqn{x \in [a,b]}
#' is:
#'
#' \deqn{f_{T}(x) = \frac{1}{\pi,\gamma};\frac{1}{1 + \bigl(\tfrac{x - \mu}{\gamma}\bigr)^2} ;\Bigg/ ; \left[F_{\mathrm{Cauchy}}!\left(\tfrac{b - \mu}{\gamma}\right) - F_{\mathrm{Cauchy}}!\left(\tfrac{a - \mu}{\gamma}\right)\right],}
#'
#' and (f_\{T\}(x)=0) for (x < a) or (x > b).
#'
#' Here \eqn{F_{\mathrm{Cauchy}}} is the standard Cauchy CDF \eqn{with
#' parameters}\\mu,\\gamma)$.
#'
#' @param loc Location parameter of the Cauchy distribution.
#' @param scale Scale parameter of the Cauchy distribution.
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
#'   \item BF TruncatedCauchy distribution object (for model building) when
#'     `sample = FALSE`.
#'   \item JAX array of samples drawn from the TruncatedCauchy distribution (for
#'     direct sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.truncated_cauchy(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#truncatedcauchy}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Cauchy_distribution "Cauchy
#' distribution"
#'
#' \[2\]: https://www.tandfonline.com/doi/full/10.1080/00207390600595223 "Full
#' article: A truncated Cauchy distribution"
#'
#' \[3\]:
#' https://rstudio.github.io/tfprobability/reference/tfd_truncated_cauchy.html
#' "The Truncated Cauchy distribution. — tfd_truncated_cauchy"
#'
#' @export
bf.dist.truncated_cauchy=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$truncated_cauchy(
       loc=.BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       low= .bf_arr(low),
       high = .bf_arr(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

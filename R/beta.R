#' @title Beta distribution
#'
#' @description
#' Samples from a Beta distribution, defined on the interval \[0, 1\]. The Beta
#' distribution is a versatile distribution often used to model probabilities or
#' proportions. It is parameterized by two positive shape parameters, usually
#' denoted \eqn{\alpha} and \eqn{\beta>0}, control the shape of the density (how
#' much mass is pushed toward 0, 1, or intermediate).
#'
#' \deqn{X \sim Beta(\alpha,\beta), \\f(x)=\frac{ x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \\ B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}, \\F(x)=I_{x}(\alpha+\beta)}
#'
#' where \eqn{B(\alpha, \beta)} is the Beta function:
#'
#' \deqn{B(\alpha, \beta) = \int_0^1 x^{\alpha - 1} (1 - x)^{\beta - 1} , dx = \frac{\Gamma(\alpha),\Gamma (\beta)}{\Gamma(\alpha + \beta)}.}
#'
#' where :num\eqn{\alpha` and}\\beta$ are the concentration parameters, and
#' \eqn{B(x, y)} is the Beta function.
#'
#' @param concentration1 The first concentration parameter (shape parameter).
#'   Must be positive.
#' @param concentration0 The second concentration parameter (shape parameter).
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
#'   \item When `sample = FALSE`: A BF Beta distribution object (for  model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     BernoulliLogits distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF Beta object (for advanced use
#'     cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.beta(concentration1 = 1.0, concentration0 = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#beta}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Beta_distribution "Beta distribution -
#' Wikipedia"
#'
#' \[2\]: https://www.statlect.com/probability-distributions/beta-distribution
#' "Beta distribution | Properties, proofs, exercises - StatLect"
#'
#' \[3\]: https://reference.wolfram.com/language/ref/BetaDistribution.html
#' "BetaDistribution - Wolfram Language Documentation"
#'
#' @export
bf.dist.beta=function(concentration1, concentration0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$beta(
       concentration1 = .BF_env$jnp$asarray(concentration1),
       concentration0 = .BF_env$jnp$asarray(concentration0),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

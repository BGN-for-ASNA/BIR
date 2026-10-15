#' @title Cauchy distribution
#'
#' @description
#' Samples from a Cauchy
#'
#' The Cauchy distribution, also known as the Lorentz distribution, is a
#' continuous probability distribution that arises frequently in various fields,
#' including physics and statistics. It is characterized by its heavy tails,
#' which extend indefinitely. This means it has a higher probability of extreme
#' values compared to the normal distribution.
#'
#' * **Parameters**:
#' * **Location parameter** (\eqn{\mu}): Determines the peak of the
#' distribution.
#' * **Scale parameter** (\eqn{\gamma}): Describes the half-width at
#' half-maximum (HWHM) of the peak.
#'
#' \deqn{f(x; \mu, \gamma) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x - \mu}{\gamma}\right)^2\right]}}
#'
#' where:
#'
#' * \eqn{x} is the random variable,
#' * \eqn{\mu} is the location parameter,
#' * \eqn{\gamma} is the scale parameter.
#'
#' @param loc Location parameter.  Defaults to 0.0.
#' @param scale See the Python documentation of `scale`.
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
#'   \item When `sample = FALSE`: A BF Cauchy distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Cauchy
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.cauchy(loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#cauchy}
#'
#' @export
bf.dist.cauchy=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$cauchy(
       loc=.BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

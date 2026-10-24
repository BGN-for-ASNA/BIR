#' @title Student's t distribution
#'
#' @description
#' The Student's t-distribution is a probability distribution that arises in
#' hypothesis testing involving the mean of a normally distributed population
#' when the population standard deviation is unknown. It is similar to the
#' normal distribution, but has heavier tails, making it more robust to
#' outliers. For large \eqn{\nu}, it converges to the Normal distribution.
#'
#' \deqn{X \sim t_\nu(\mu, \sigma)}
#'
#' where:
#'
#' * \eqn{\mu} is the **location (mean)** parameter
#' * \eqn{\sigma > 0} is the **scale** parameter
#' * \eqn{\nu > 0} is the **degrees of freedom** controlling the tail heaviness
#'
#' The probability density function (pdf) of the Student's t distribution is:
#'
#' \deqn{f(x \mid \nu, \mu, \sigma) = \frac{\Gamma!\left(\frac{\nu+1}{2}\right)} {\Gamma!\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\sigma} \left[1 + \frac{1}{\nu} \left(\frac{x - \mu}{\sigma}\right)^2 \right]^{-\frac{\nu+1}{2}}}
#'
#' * The heavier tails (for small \eqn{ν}) allow for larger outliers.
#' * For \eqn{\nu \to \infty}, this approaches \eqn{\mathcal{N}(\mu, \sigma^2)}.
#'
#' @param df Degrees of freedom, must be positive.
#' @param loc Location parameter, defaults to 0.0.
#' @param scale Scale parameter, defaults to 1.0.
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
#'   \item When `sample = FALSE`: A BF StudentT distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the StudentT
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.student_t(df = 2, loc = 0.0, scale = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#studentt}
#'
#' @references
#' [Wikipedia – Student’s t distribution:
#' PDF](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Probability_density_function)
#'
#' @export
bf.dist.student_t=function(df, loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$student_t(
       df = .BF_env$jnp$asarray((df)),
       loc= .BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

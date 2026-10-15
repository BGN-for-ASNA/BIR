#' @title Pareto distribution
#'
#' @description
#' Samples from a Pareto distribution.
#'
#' The **Pareto distribution**, named after economist Vilfredo Pareto, is a
#' **power-law** probability distribution used to describe phenomena with
#' “rich-get-richer” or “heavy-tail” properties — for example, income
#' distribution, city sizes, or wealth concentration. It is characterized by:
#'
#' * a **scale parameter** \eqn{x_m > 0} (the minimum possible value), and
#' * a **shape parameter** \eqn{\alpha > 0} (which controls the tail heaviness).
#'
#' A random variable \eqn{X \sim \text{Pareto}(\alpha, x_m)} takes values \eqn{x
#' \ge x_m}.
#'
#' \deqn{f(x \mid \alpha, x_m) = \begin{cases} \dfrac{\alpha x_m^\alpha}{x^{\alpha + 1}}, & x \ge x_m, \ 0, & x < x_m. \end{cases}}
#'
#' @param scale Scale parameter of the Pareto distribution. Must be positive.
#' @param alpha Shape parameter of the Pareto distribution. Must be positive.
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
#'   \item When `sample = FALSE`: A BF Pareto distribution object (for model
#'   \item building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Pareto
#'   \item distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'   \item use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.pareto(scale = 2.0, alpha = 3.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#pareto}
#'
#' @references
#' [Wikipedia: Pareto distribution –
#' PDF](https://en.wikipedia.org/wiki/Pareto_distribution#Probability_density_function)
#'
#' Arnold, B. C. (2015). *Pareto Distributions*, Second Edition. CRC Press.
#'
#' @export
bf.dist.pareto=function(scale, alpha, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$pareto(
       scale = .BF_env$jnp$asarray(scale),
       alpha = .BF_env$jnp$asarray(alpha),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

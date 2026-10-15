#' @title Dirichlet distribution
#'
#' @description
#' Samples from a Dirichlet distribution.
#'
#' The Dirichlet distribution is a multivariate generalization of the Beta
#' distribution. It is a probability distribution over a simplex, which is a set
#' of vectors where each element is non-negative and sums to one. It is often
#' used as a prior distribution for categorical distributions.
#'
#' For ( X = (X_1, \\dots, X_K) \\sim \\mathrm\{Dir\}(\\alpha)), the density is
#'
#' \deqn{f( x; \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}}
#'
#' for ( x) in the simplex, and 0 otherwise. Here (B(\\alpha)) is the
#' multivariate Beta function (the normalization constant):
#'
#' \deqn{B(\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^K \alpha_i\right)}.}
#'
#' @param concentration The concentration parameter(s) of the Dirichlet
#'   distribution. Must be a positive array.
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
#'   \item When `sample = FALSE`: A BF Dirichlet Distribution distribution object
#'     (for  model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Dirichlet
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF Dirichlet Distribution object (for
#'     advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.dirichlet(concentration = jnp$array(c(1.0, 1.0, 1.0)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#dirichlet}
#'
#' @references
#' https://distribution-explorer.github.io/multivariate_continuous/dirichlet.html
#'
#' https://en.wikipedia.org/wiki/Dirichlet_distribution
#'
#' @export
bf.dist.dirichlet=function(concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$dirichlet(
       concentration = .BF_env$jnp$asarray(concentration),
       validate_args = validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Beta Proportion distribution
#'
#' @description
#' The Beta Proportion distribution is a reparameterization of the conventional
#' Beta distribution in terms of a the variate mean and a precision parameter.
#' It's useful for modeling rates and proportions. It’s essentially the same
#' family as the standard Beta((\\alpha,\\beta)), but the mapping is:
#'
#' \deqn{\alpha = \mu , \kappa,\quad \beta = (1 - \mu), \kappa.}
#'
#' Thus the pdf is:
#'
#' \deqn{f(\theta \mid \mu, \kappa) = \frac{1}{B(\mu\kappa,,(1-\mu)\kappa)} ; \theta^{\mu\kappa - 1} ; (1 - \theta)^{(1 - \mu)\kappa - 1}, \\ \theta \in (0,1).}
#'
#' This parameterization is useful in hierarchical models, since one can put
#' hyperpriors on (\\mu) (the “center”) and (\\kappa) (the “spread”).
#'
#' @param mean The mean of the BetaProportion distribution,must be between 0 and
#'   1.
#' @param concentration The concentration parameter of the BetaProportion
#'   distribution.
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
#'   \item When `sample = FALSE`: A BF BetaProportion distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     BetaProportion distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF BetaProportion distribution object
#'     (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' samples <- bf.dist.beta_proportion(
#'     mean = 0.5,
#'     concentration = 2.0,
#'     sample = TRUE,
#'     shape = c(1000L)
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#beta_proportion}
#'
#' @references
#' https://mc-stan.org/docs/2_19/functions-reference/beta-proportion-distribution.html
#' "19.2 Beta Proportion Distribution | Stan Functions Reference"
#'
#' @export
bf.dist.beta_proportion=function(mean, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$beta_proportion(
       mean = .BF_env$jnp$asarray(mean),
       concentration  = .BF_env$jnp$asarray(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Zero Sum Normal distribution
#'
#' @description
#' A **zero-sum normal**  is a variant of a multivariate normal in which one (or
#' more) linear constraint(s) force certain components to **sum to zero**. In
#' practice, it’s used to model vectors of random effects (e.g. in hierarchical
#' models) where the effects are constrained to sum to zero (to avoid
#' overparameterization or enforce identifiability).
#'
#' \deqn{ZSN(\sigma) = N(0, \sigma^2 (I - \tfrac{1}{n}J)) \\ \text{where} \ ~ J_{ij} = 1 \ ~ \text{and} \\ n = \text{number of zero-sum axes}}
#'
#' where \eqn{J}is the all-ones matrix and \eqn{n} is the number of constrained
#' elements. So the covariance is \eqn{\sigma^2 (I - \tfrac{1}{n} J)}, which
#' ensures that the sum of the components is 0 (since \eqn{J \mathbf{x} =
#' \mathbf{1} \sum_i x_i}).
#'
#' @param scale Standard deviation of the underlying normal distribution before
#'   the zerosum constraint is enforced.
#' @param event_shape Shape of the event dimension of the distribution.
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
#'   \item When `sample = FALSE`: A BF ZeroSumNormal distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     ZeroSumNormal distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.zero_sum_normal(scale = 1.0, event_shape = c(2L), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zerosumnormal}
#'
#' @references
#' https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.ZeroSumNormal.html
#'
#' @export
bf.dist.zero_sum_normal=function(scale, event_shape, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event_shape=do.call(tuple, as.list(as.integer(event_shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$zero_sum_normal(
       scale = .BF_env$jnp$asarray(scale),
       event_shape = event_shape,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

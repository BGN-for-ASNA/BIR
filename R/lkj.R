#' @title Lewandowski Kurowicka Joe (LKJ) distribution
#'
#' @description
#' The LKJ distribution is controlled by the concentration parameter \eqn{\eta}
#' to make the probability of the correlation matrix \eqn{M` proportional
#' to}\\det(M)^\{\\eta - 1\}\eqn{. When}\\eta = 1$, the distribution is uniform
#' over correlation matrices.  When \eqn{\eta > 1}, the distribution favors
#' samples with large determinants. When \eqn{\eta < 1}, the distribution favors
#' samples with small determinants.
#'
#' \deqn{P(M) \propto |\det(M)|^{\eta - 1}}
#'
#' @param dimension The dimension of the correlation matrices.
#' @param concentration The concentration/shape parameter of the distribution
#'   (often referred to as eta). Must be positive.
#' @param sample_method Either “cvine” or “onion”. Both methods are proposed in
#'   \[1\] and offer the same distribution over correlation matrices. But they are
#'   different in how to generate samples. Defaults to “onion”.
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
#'   \item When `sample = FALSE`: A BF LKJ distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the LKJ
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.lkj(dimension = 2L, concentration = 1.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#lkj}
#'
#' @references
#' https://en.wikipedia.org/wiki/Kumaraswamy_distribution
#'
#' @export
bf.dist.lkj=function(dimension, concentration=1.0, sample_method='onion', validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     dimension=as.integer(dimension);
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$lkj(
       dimension = dimension,
       concentration= .BF_env$jnp$asarray(concentration),
       sample_method= sample_method,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

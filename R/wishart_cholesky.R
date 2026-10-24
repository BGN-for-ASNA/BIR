#' @title Wishart Cholesky distribution
#'
#' @description
#' * The **Wishart** distribution is a distribution over positive definite
#' matrices, often used as a prior for covariance or precision matrices in
#' multivariate normal models.
#' * The **Cholesky parameterization** of the Wishart (called “wishart_cholesky”
#' in Stan, for example) reparameterizes the Wishart over its **lower (or upper)
#' triangular Cholesky factor**. This is useful for numerical stability and
#' unconstrained parameterization in Bayesian sampling frameworks.
#' * In this parameterization, one works with a lower‐triangular matrix
#' \eqn{L_W} such that
#'
#' \deqn{\Sigma = L_W L_W^\top}
#'
#' and imposes a density over \eqn{L_W} corresponding to the induced Wishart
#' density on (\\Sigma).
#'
#' The probability density function (PDF) is given by:
#'
#' If the dimension is \eqn{K}, degrees of freedom \eqn{\nu > K - 1}, and
#' \eqn{L_S} is the lower‐triangular Cholesky factor of the scale matrix
#' \eqn{S}, then the log density for \eqn{L_W} under the **wishart_cholesky**
#' distribution is:
#'
#' \deqn{\log p(L_W \mid \nu, L_S) = \log \bigl[ \text{Wishart}(L_W L_W^\top \mid \nu, L_S L_S^\top) \bigr] + \log \bigl| J_{f^{-1}} \bigr|}
#'
#' where \eqn{J_{f^{-1}}} is the Jacobian of the transformation from \eqn{L_W}
#' to \eqn{\Sigma = L_W L_W^\top}.
#'
#' The Jacobian term (absolute log determinant) is:
#'
#' \deqn{\log |J_{f^{-1}}| = K \log 2 ; \sum_{k=1}^K (K - k + 1),\log L_{W_{k,k}}.}
#'
#' Thus the density includes both the usual Wishart density on \eqn{\Sigma} plus
#' this extra Jacobian (which penalizes or weights the diagonal entries of
#' \eqn{L_W}).
#'
#' @param concentration See the Python documentation of `concentration`.
#' @param scale_matrix See the Python documentation of `scale_matrix`.
#' @param rate_matrix See the Python documentation of `rate_matrix`.
#' @param scale_tril See the Python documentation of `scale_tril`.
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
#' @return A random variable (model building) or draws (`sample = TRUE`).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.wishart_cholesky(concentration = 5.0, scale_matrix = jnp$eye(2L), sample = TRUE)
#' }
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Wishart_distribution "Wishart
#' distribution"
#'
#' \[2\]:
#' https://mc-stan.org/docs/2_32/functions-reference/wishart-cholesky-distribution.html"28.2
#' Wishart distribution, Cholesky Parameterization - Stan"
#'
#' @export
bf.dist.wishart_cholesky=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(scale_matrix)){scale_matrix = .BF_env$jnp$asarray(scale_matrix)}
     if(!.BF_env$.py$is_none(rate_matrix)){rate_matrix = .BF_env$jnp$asarray(rate_matrix)}
     if(!.BF_env$.py$is_none(scale_tril)){scale_tril = .BF_env$jnp$asarray(scale_tril)}
     .BF_env$.bf_instance$dist$wishart_cholesky(
       concentration  = .BF_env$jnp$asarray(concentration),
       scale_matrix= scale_matrix,
       rate_matrix= rate_matrix,
       scale_tril= scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}


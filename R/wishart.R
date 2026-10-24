#' @title Wishart distribution
#'
#' @description
#' The Wishart distribution is a multivariate distribution used to model
#' positive definite matrices, often representing covariance matrices. It's
#' commonly used in Bayesian statistics and machine learning, particularly in
#' models involving covariance estimation.
#'
#' \deqn{\mathbf{X} \sim \text{Wishart}(\text{scale_matrix}=\mathbf{V}, \text{concentration}=n)}
#'
#' where:
#'
#' * (p) = dimension of the square matrices
#' * (\\mathbf\{V\} \\in \\mathbb\{R\}^\{p \\times p\}) = positive definite
#' **scale matrix** (`scale_matrix`)
#' * (n \\ge p) = **degrees of freedom** (`concentration`)
#'
#' The Wishart distribution is the **multivariate generalization of the
#' chi-squared distribution**. It is commonly used as the distribution of the
#' **sample covariance matrix** for multivariate normal samples.
#'
#' \deqn{f(\mathbf{X} \mid \mathbf{V}, n) = \frac{ |\mathbf{X}|^{(n-p-1)/2} \exp\big(-\frac{1}{2} \mathrm{tr}(\mathbf{V}^{-1} \mathbf{X}) \big) } { 2^{np/2} |\mathbf{V}|^{n/2} \Gamma_p(n/2) }, \quad \mathbf{X} \succ 0}
#'
#' where:
#'
#' * \eqn{|\mathbf{X}|} = determinant of \eqn{\mathbf{X}}
#' * \eqn{\mathrm{tr}(\cdot)} = trace
#' * \eqn{\Gamma_p(\cdot)} = multivariate Gamma function:
#'
#' @param concentration Positive concentration parameter analogous to the
#'   concentration of a Gamma distribution. The concentration must be larger than
#'   the dimensionality of the scale matrix.
#' @param scale_matrix Scale matrix analogous to the inverse rate of a Gamma
#'   distribution.
#' @param rate_matrix Rate matrix anaologous to the rate of a Gamma
#'   distribution.
#' @param scale_tril Cholesky decomposition of the :code:`scale_matrix`.
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
#'   \item When `sample = FALSE`: A BF Wishart distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Wishart
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.wishart(concentration = 5.0, scale_matrix = jnp$eye(2L), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#wishart}
#'
#' @references
#' [Wikipedia – Wishart
#' distribution](https://en.wikipedia.org/wiki/Wishart_distribution)
#'
#' @export
bf.dist.wishart=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!.BF_env$.py$is_none(scale_matrix)){scale_matrix = .BF_env$jnp$asarray(scale_matrix)}
     if(!.BF_env$.py$is_none(rate_matrix)){rate_matrix = .BF_env$jnp$asarray(rate_matrix)}
     if(!.BF_env$.py$is_none(scale_tril)){scale_tril = .BF_env$jnp$asarray(scale_tril)}
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$wishart(
       concentration = .BF_env$jnp$asarray(concentration),
       scale_matrix= scale_matrix,
       rate_matrix= rate_matrix,
       scale_tril= scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

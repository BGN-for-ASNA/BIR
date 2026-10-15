#' @title Wishart distribution for covariance matrices.
#' @description
#' The Wishart distribution is a multivariate distribution used to model positive definite matrices,
#' often representing covariance matrices. It's commonly used in Bayesian statistics and machine learning,
#' particularly in models involving covariance estimation.
#'
#' @param concentration A positive concentration parameter analogous to the
#'   concentration of a Gamma distribution. The concentration must be larger
#'   than the dimensionality of the scale matrix.
#' @param scale_matrix A scale matrix analogous to the inverse rate of a Gamma
#'   distribution.
#' @param rate_matrix A rate matrix anaologous to the rate of a Gamma
#'   distribution.
#' @param scale_tril Cholesky decomposition of the `scale_matrix`.
#' @param shape A numeric vector specifying the shape. When `sample=False`
#'   (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions
#'   (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj If `TRUE`, returns the raw BF distribution object
#'   instead of creating a sample site. This is essential for building complex
#'   distributions like `MixtureSameFamily`.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BF Wishart distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Wishart distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#wishart}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.wishart(concentration = 5, scale_matrix = matrix(c(1,0,0,1), nrow = 2), sample = TRUE)
#' }
#' @export
bf.dist.wishart=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
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

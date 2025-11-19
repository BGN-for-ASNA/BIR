#' @title Wishart Cholesky Distribution
#' @description
#' The Wishart distribution is a multivariate distribution used as a prior distribution
#' for covariance matrices. This implementation represents the distribution in terms
#' of its Cholesky decomposition.
#'
#' @title WishartCholesky Distribution
#' @description The Wishart distribution is a multivariate distribution used as a prior distribution
#' for covariance matrices. This implementation represents the distribution in terms
#' of its Cholesky decomposition.
#'
#' @param concentration (numeric or vector) Positive concentration parameter analogous to the
#'   concentration of a `Gamma` distribution. The concentration must be larger
#'   than the dimensionality of the scale matrix.
#' @param scale_matrix (numeric vector, matrix, or array, optional) Scale matrix analogous to the inverse rate of a `Gamma`
#'   distribution. If not provided, `rate_matrix` or `scale_tril` must be.
#' @param rate_matrix (numeric vector, matrix, or array, optional) Rate matrix anaologous to the rate of a `Gamma`
#'   distribution. If not provided, `scale_matrix` or `scale_tril` must be.
#' @param scale_tril (numeric vector, matrix, or array, optional) Cholesky decomposition of the `scale_matrix`.
#'   If not provided, `scale_matrix` or `rate_matrix` must be.
#' @param shape A numeric vector.  This is used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Wishart Cholesky  distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Wishart Cholesky  distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.wishart_cholesky(
#' concentration = 5,
#' scale_matrix = matrix(c(1,0,0,1),
#' nrow = 2),
#' sample = TRUE)
#' }
#' @export
bi.dist.wishart_cholesky=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BI_env$.py$is_none(scale_matrix)){scale_matrix = .BI_env$jnp$array(scale_matrix)}
     if(!.BI_env$.py$is_none(rate_matrix)){rate_matrix = .BI_env$jnp$array(rate_matrix)}
     if(!.BI_env$.py$is_none(scale_tril)){scale_tril = .BI_env$jnp$array(scale_tril)}
     .BI_env$.bi_instance$dist$wishart_cholesky(
       concentration  = .BI_env$jnp$array(concentration),
       scale_matrix= scale_matrix,
       rate_matrix= rate_matrix,
       scale_tril= scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}


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
#' @return
#'  - When \code{sample=FALSE}, a BI Wishart Cholesky  distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Wishart Cholesky  distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.wishart_cholesky(sample = TRUE)
#' }
#' @export
bi.dist.wishart_cholesky=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed)
     if(!py$is_none(scale_matrix)){scale_matrix = jnp$array(scale_matrix)}
     if(!py$is_none(rate_matrix)){rate_matrix = jnp$array(rate_matrix)}
     if(!py$is_none(scale_tril)){scale_tril = jnp$array(scale_tril)}
     .bi$dist$wishart_cholesky(
       concentration  = jnp$array(concentration),
       scale_matrix= scale_matrix,
       rate_matrix= rate_matrix,
       scale_tril= scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}


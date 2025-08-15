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
#' @param create_obj If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site. This is essential for building complex
#'   distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Wishart distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Wishart distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#wishart}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.wishart(concentration = 5, scale_matrix = matrix(c(1,0,0,1), nrow = 2), sample = TRUE)
#' }
#' @export
bi.dist.wishart=function(concentration, scale_matrix=py_none(), rate_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!py$is_none(scale_matrix)){scale_matrix = jnp$array(scale_matrix)}
     if(!py$is_none(rate_matrix)){rate_matrix = jnp$array(rate_matrix)}
     if(!py$is_none(scale_tril)){scale_tril = jnp$array(scale_tril)}
     seed=as.integer(seed);
     .bi$dist$wishart(
       concentration = jnp$array(concentration),
       scale_matrix= scale_matrix,
       rate_matrix= rate_matrix,
       scale_tril= scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

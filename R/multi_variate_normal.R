#' @title Samples from a Multivariate Normal distribution.
#'
#' @description The Multivariate Normal distribution, also known as the Gaussian distribution in multiple dimensions,
#' is a probability distribution that arises frequently in statistics and machine learning. It is
#' defined by its mean vector and covariance matrix, which describe the central tendency and
#' spread of the distribution, respectively.
#'
#' \deqn{p(x) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)}
#'
#' where:
#' - \eqn{x} is a \eqn{n}-dimensional vector of random variables.
#' - \eqn{\mu} is the mean vector.
#' - \eqn{\Sigma} is the covariance matrix.
#'
#' @export
#' @importFrom reticulate py_none tuple
#' @param loc A numeric vector representing the mean vector of the distribution.
#' @param covariance_matrix A numeric vector, matrix, or array representing the covariance matrix of the distribution. Must be positive definite.
#' @param precision_matrix A numeric vector, matrix, or array representing the precision matrix (inverse of the covariance matrix) of the distribution. Must be positive definite.
#' @param scale_tril A numeric vector, matrix, or array representing the lower triangular Cholesky decomposition of the covariance matrix.
#' @param shape A numeric vector representing the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector representing an optional boolean array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'  - When \code{sample=FALSE}, a BI Multivariate Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Multivariate Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multivariate-normal}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.multivariate_normal(
#' loc =  c(1.0, 0.0, -2.0),
#' covariance_matrix = matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5), nrow = 3, byrow = TRUE),
#' sample = TRUE)
#' }
#' @export
bi.dist.multivariate_normal=function(loc=0.0, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!py$is_none(covariance_matrix)){covariance_matrix = jnp$array(covariance_matrix)}
     if(!py$is_none(precision_matrix)){precision_matrix = jnp$array(precision_matrix)}
     if(!py$is_none(scale_tril)){scale_tril = jnp$array(scale_tril)}

     seed=as.integer(seed);
     .bi$dist$multivariate_normal(
       loc = jnp$array(loc),
       covariance_matrix = covariance_matrix,
       precision_matrix = precision_matrix,
       scale_tril = scale_tril,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

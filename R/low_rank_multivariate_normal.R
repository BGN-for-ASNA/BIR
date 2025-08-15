#' @title Low Rank Multivariate Normal Distribution
#'
#' @description Represents a multivariate normal distribution with a low-rank covariance structure.
#'
#' The probability density function (PDF) is:
#' \deqn{p(x) = \frac{1}{\sqrt{(2\pi)^K |\Sigma|}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)}
#'
#' where:
#'
#' * \eqn{x} is a vector of observations.
#' * \eqn{\mu} is the mean vector.
#' * \eqn{\Sigma} is the covariance matrix, represented in a low-rank form.
#'
#' @param loc A numeric vector representing the mean vector.
#' @param cov_factor A numeric vector or matrix used to construct the covariance matrix.
#' @param cov_diag A numeric vector representing the diagonal elements of the covariance matrix.
#' @return
#'  - When \code{sample=FALSE}, a BI Low Rank Multivariate Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Low Rank Multivariate Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lowrankmultivariatenormal}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' event_size = 10
#' rank = 5
#' bi.dist.low_rank_multivariate_normal(
#'   loc = bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)*2,
#'   cov_factor = bi.dist.normal(0,1,shape = c(event_size, rank), sample = TRUE),
#'   cov_diag = jnp$exp(bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)),
#'   sample = TRUE)
#' }
#' @export
bi.dist.low_rank_multivariate_normal=function(loc, cov_factor, cov_diag, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$low_rank_multivariate_normal(
       loc = jnp$array(loc),
       cov_factor = jnp$array(cov_factor),
       cov_diag = jnp$array(cov_diag),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

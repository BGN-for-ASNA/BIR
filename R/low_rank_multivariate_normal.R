#' @title Low Rank Multivariate Normal Distribution
#'
#' @description
#' The *Low-Rank Multivariate Normal* (LRMVN) distribution is a parameterizaton of the multivariate normal distribution where the covariance matrix is expressed as a low-rank plus diagonal decomposition:
#'\deqn{
#'   \Sigma = F F^\top + D
#'}
#' where $F$ is a low-rank matrix (capturing correlations) and $D$ is a diagonal matrix (capturing independent noise).
#' This representation is often used in probabilistic modeling and variational inference to efficiently handle high-dimensional
#' Gaussian distributions with structured covariance.
#'
#' @param loc A numeric vector representing the mean vector.
#' @param cov_factor A numeric vector or matrix used to construct the covariance matrix.
#' @param cov_diag A numeric vector representing the diagonal elements of the covariance matrix.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param shape Numeric vector. A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

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
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' event_size = 10
#' rank = 5
#' bi.dist.low_rank_multivariate_normal(
#'   loc = bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)*2,
#'   cov_factor = bi.dist.normal(0,1,shape = c(event_size, rank), sample = TRUE),
#'   cov_diag = bi.dist.normal(10,0.5,shape = c(event_size), sample = TRUE),
#'   sample = TRUE)
#' }
#' @export
bi.dist.low_rank_multivariate_normal=function(loc, cov_factor, cov_diag, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$low_rank_multivariate_normal(
       loc = .BI_env$jnp$array(loc),
       cov_factor = .BI_env$jnp$array(cov_factor),
       cov_diag = .BI_env$jnp$array(cov_diag),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

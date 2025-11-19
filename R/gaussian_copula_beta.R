#' @title Gaussian Copula Beta distribution.
#' @description This distribution combines a Gaussian copula with a Beta distribution.
#' The Gaussian copula models the dependence structure between random variables,
#' while the Beta distribution defines the marginal distributions of each variable.
#'
#' @param concentration1 A numeric vector or matrix representing the first shape parameter of the Beta distribution.
#' @param concentration0 A numeric vector or matrix representing the second shape parameter of the Beta distribution.
#' @param correlation_matrix array_like, optional: Correlation matrix of the coupling multivariate normal distribution. Defaults to `reticulate::py_none()`.
#' @param correlation_cholesky A numeric vector, matrix, or array representing the Cholesky decomposition of the correlation matrix.
#' @param shape A numeric vector.  This is used as `sample_shape` to draw a raw JAX array of the given shape when `sample=True`.
#' @param event Integer indicating the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BI Gaussian Copula Beta distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Gaussian Copula Beta distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#gaussiancopulabetadistribution}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.gaussian_copula_beta(
#'   concentration1 = c(2.0, 3.0),
#'   concentration0 = c(5.0, 3.0),
#'   correlation_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
#'   sample = TRUE)
#'   }
#' @export
bi.dist.gaussian_copula_beta=function(concentration1, concentration0, correlation_matrix=py_none(), correlation_cholesky=py_none(), validate_args=FALSE, name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     return(print("No more available since jax > 0.06"))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BI_env$.py$is_none(correlation_cholesky)){correlation_cholesky = .BI_env$jnp$array(correlation_cholesky)}
     if(!.BI_env$.py$is_none(correlation_matrix)){correlation_matrix = .BI_env$jnp$array(correlation_matrix)}

     .BI_env$.bi_instance$dist$gaussian_copula_beta(
       concentration1 = .BI_env$jnp$array(concentration1),
       concentration0 = .BI_env$jnp$array(concentration0),
       correlation_matrix = correlation_matrix,
       correlation_cholesky = correlation_cholesky,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

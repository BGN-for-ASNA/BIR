#' @title Matrix Normal Distribution
#'
#' @description
#' Samples from a Matrix Normal distribution, which is a multivariate normal distribution over matrices.
#' The distribution is characterized by a location matrix and two lower triangular matrices that define the correlation structure.
#' The distribution is related to the multivariate normal distribution in the following way.
#' If \deqn{X ~ MN(loc,U,V)} then \deqn{vec(X) ~ MVN(vec(loc), kron(V,U) }.
#'
#' \deqn{p(x) = \frac{1}{2\pi^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)}
#'
#' @param loc A numeric vector, matrix, or array representing the location of the distribution.
#' @param scale_tril_row A numeric vector, matrix, or array representing the lower cholesky of rows correlation matrix.
#' @param scale_tril_column A numeric vector, matrix, or array representing the lower cholesky of columns correlation matrix.
#' @param shape A numeric vector specifying the shape of the distribution.  Must be a vector.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A logical vector, matrix, or array (.BI_env$jnp$array) to mask observations.
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
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Matrix Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Matrix Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' n_rows= 3
#' n_cols = 4
#' loc = matrix(rep(0,n_rows*n_cols), nrow = n_rows, ncol = n_cols,byrow = TRUE)
#'
#' U_row_cov =
#' matrix(c(1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0),
#' nrow = n_rows, ncol = n_rows,byrow = TRUE)
#' scale_tril_row = chol(U_row_cov)
#'
#' V_col_cov = matrix(c(2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1,
#' 0.2, 2.0, 0.0, 0.4, -0.2, 0.0, 2.0),
#' nrow = n_cols, ncol = n_cols,byrow = TRUE)
#' scale_tril_column = chol(V_col_cov)
#'
#'
#' bi.dist.matrix_normal(
#' loc = loc,
#' scale_tril_row = scale_tril_row,
#' scale_tril_column = scale_tril_column,
#' sample = TRUE
#' )
#' }
#' @export

bi.dist.matrix_normal=function(loc, scale_tril_row, scale_tril_column, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$matrix_normal(
       loc = .BI_env$jnp$array(loc),
       scale_tril_row = .BI_env$jnp$array(scale_tril_row),
       scale_tril_column = .BI_env$jnp$array(scale_tril_column),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

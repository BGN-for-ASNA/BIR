#' @title Conditional Autoregressive (CAR) Distribution
#'
#' @description The CAR distribution models a vector of variables where each variable is a linear
#' combination of its neighbors in a graph.
#'
#' \deqn{p(x) = \prod_{i=1}^{K} \mathcal{N}(x_i | \mu_i, \Sigma_i)}
#'
#' where \eqn{\mu_i} is a function of the values of the neighbors of site \eqn{i}
#' and \eqn{\Sigma_i} is the variance of site \eqn{i}.
#'
#' The CAR distribution is a special case of the multivariate normal distribution.
#' It is used to model spatial data, such as temperature or precipitation.
#'
#' @param loc Numeric vector, matrix, or array representing the mean of the distribution.
#' @param correlation Numeric vector, matrix, or array representing the correlation between variables.
#' @param conditional_precision Numeric vector, matrix, or array representing the precision of the distribution.
#' @param adj_matrix Numeric vector, matrix, or array representing the adjacency matrix defining the graph.
#' @param is_sparse Logical indicating whether the adjacency matrix is sparse. Defaults to `FALSE`.
#' @param validate_args Logical indicating whether to validate arguments. Defaults to `reticulate::py_none()`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI CAR distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the CAR distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).

#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.car(
#'   loc = c(1.,2.),
#'   correlation = 0.9,
#'   conditional_precision = 1.,
#'   adj_matrix = matrix(c(1,0,0,1), nrow = 2),
#'   sample = TRUE
#'  )
#' }
#' @export
#'
bi.dist.car=function(loc, correlation, conditional_precision, adj_matrix, is_sparse=FALSE, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     if(!py$is_none(correlation)){correlation = jnp$array(correlation)}
     if(!py$is_none(conditional_precision)){conditional_precision = jnp$array(conditional_precision)}
     if(!py$is_none(adj_matrix)){adj_matrix = jnp$array(adj_matrix)}
     .bi$dist$car(
       loc = jnp$array(loc),
       correlation=correlation,
       conditional_precision=conditional_precision,
       adj_matrix=adj_matrix,
       is_sparse= is_sparse,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Conditional Autoregressive (CAR) Distribution
#'
#' @description The CAR distribution models a vector of variables where each variable is a linear
#' combination of its neighbors in a graph. The CAR model captures spatial dependence in areal data by modeling each observation as conditionally dependent on its neighbors.It specifies a joint distribution of a vector of random variables $\mathbf{y} = (y_1, y_2, \dots, y_N)$ based on their conditional distributions, where each $y_i$ is conditionally independent of all other variables given its neighbors.
#' * **Application**: Widely used in disease mapping, environmental modeling, and spatial econometrics to account for spatial autocorrelation.
#'
#' The CAR distribution is a special case of the multivariate normal distribution.
#' It is used to model spatial data, such as temperature or precipitation.
#'
#' @param loc Numeric vector, matrix, or array representing the mean of the distribution.
#' @param correlation Numeric vector, matrix, or array representing the correlation between variables.
#' @param conditional_precision Numeric vector, matrix, or array representing the precision of the distribution.
#' @param adj_matrix Numeric vector, matrix, or array representing the adjacency matrix defining the graph.
#' @param is_sparse Logical indicating whether the adjacency matrix is sparse. Defaults to `FALSE`.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the
#'   given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution
#'   object instead of creating a sample site. This is essential for
#'   building complex distributions like `MixtureSameFamily`.
#' @param validate_args Logical indicating whether to validate arguments. Defaults to `reticulate::py_none()`.
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
#'  - When \code{sample=FALSE}, a BI CAR distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the CAR distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).

#' @examples
#' \donttest{
#' library(BayesianInference)
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
bi.dist.car=function(loc, correlation, conditional_precision, adj_matrix, is_sparse=FALSE, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BI_env$.py$is_none(correlation)){correlation = .BI_env$jnp$array(correlation)}
     if(!.BI_env$.py$is_none(conditional_precision)){conditional_precision = .BI_env$jnp$array(conditional_precision)}
     if(!.BI_env$.py$is_none(adj_matrix)){adj_matrix = .BI_env$jnp$array(adj_matrix)}
     .BI_env$.bi_instance$dist$car(
       loc = .BI_env$jnp$array(loc),
       correlation=correlation,
       conditional_precision=conditional_precision,
       adj_matrix=adj_matrix,
       is_sparse= is_sparse,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title LKJ Cholesky Distribution
#'
#' @description
#' The LKJ Cholesky distribution is a family of distributions
#' on symmetric matrices, often used as a prior for the Cholesky decomposition of a
#' symmetric matrix. It is particularly useful in Bayesian inference for models with
#' covariance structure.
#'
#' @name bi.dist.lkj_cholesky
#'
#' @param dimension Numeric for the dimensions of the LKJ Cholesky matrix.
#' @param concentration Numeric. A parameter controlling the concentration of the distribution
#'   around the identity matrix. Higher values indicate greater concentration.
#'   Must be greater than 1.
#' @param sample_method onion
#' @param validate_args None
#' @param shape Numeric vector; A multi-purpose argument for shaping. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array
#'   of the given shape.
#' @param event Numeric; The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector; Optional boolean array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample
#'   site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'    - When \code{sample=FALSE}: A BI LKJ Cholesky distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the LKJ Cholesky distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m <- importBI(platform='cpu')
#' bi.dist.lkj_cholesky(dimension = 2, concentration = 1., sample = TRUE)
#' }
#'
#' @export
bi.dist.lkj_cholesky <- function(dimension, concentration=1.0, sample_method='onion', validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
  shape <- do.call(tuple, as.list(as.integer(shape)))
  dimension <- as.integer(dimension)
  seed <- as.integer(seed)
  .bi$dist$lkj_cholesky(
    dimension = jnp$array(dimension),
    concentration = jnp$array(concentration),
    sample_method = sample_method,
    validate_args = validate_args,
    name = name,
    obs = obs,
    mask = mask,
    sample = sample,
    seed = seed,
    shape = shape,
    event = event,
    create_obj = create_obj
  )
}

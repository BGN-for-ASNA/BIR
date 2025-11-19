#' @title LKJ Cholesky Distribution
#'
#' @description Samples from an LKJ (Lewandowski, Kurowicka, Joe) distribution for correlation matrices.
#'
#'   The LKJ distribution is controlled by the concentration parameter `eta` to make the probability of the correlation matrix M proportional to `det(M)^(eta - 1)`.
#'   When `eta = 1`, the distribution is uniform over correlation matrices.
#'   When `eta > 1`, the distribution favors samples with large determinants.
#'   When `eta < 1`, the distribution favors samples with small determinants.
#'
#' @param dimension An integer representing the dimension of the correlation
#'   matrices.
#' @param concentration A numeric vector representing the concentration/shape parameter of the distribution (often referred to as eta). Must be positive.
#' @param sample_method String: Either "cvine" or "onion". Methods proposed offer the same distribution over correlation matrices but differ in how to generate samples. Defaults to "onion".
#' @param shape A numeric vector used for shaping. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE`, it is used as `sample_shape`.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values. Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`, samples are drawn directly. If `FALSE`, a random variable is created. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when `sample = TRUE`. Defaults to 0.
#' @param obs A numeric vector or array of observed values. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable. Defaults to 'x'.
#' @param to_jax Logical. Defaults to TRUE.
#'
#' @return
#'    - When `sample=FALSE`: A BI LKJ distribution object.
#'
#'    - When `sample=TRUE`: A JAX array of samples.
#'
#'    - When `create_obj=TRUE`: The raw BI distribution object.
#'
#' @seealso Wrapper of \url{https://num.pyro.ai/en/stable/distributions.html#lkj}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.lkj(dimension = 2, concentration=1.0, shape = c(1), sample = TRUE)
#' }
#' @export

bi.dist.laplace=function(
    loc=0.0,
    scale=1.0,
    validate_args=py_none(),
    name='x',
    obs=py_none(),
    mask=py_none(),
    sample=FALSE,
    seed = py_none(),
    shape=c(),
    event=0,
    create_obj=FALSE,
    to_jax = TRUE
    ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$laplace(
       loc = .BI_env$jnp$array(loc),
       scale = .BI_env$jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

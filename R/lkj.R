#' @title Samples from an LKJ (Lewandowski, Kurowicka, Joe) distribution for correlation matrices.
#'
#' @description The LKJ distribution is controlled by the concentration parameter \eqn{\eta} to make the
#' probability of the correlation matrix \eqn{M} proportional to \eqn{\det(M)^{\eta - 1}}.
#' When \eqn{\eta = 1}, the distribution is uniform over correlation matrices.
#' When \eqn{\eta > 1}, the distribution favors samples with large determinants.
#' When \eqn{\eta < 1}, the distribution favors samples with small determinants.
#'
#' \deqn{ P(M) \propto |\det(M)|^{\eta - 1}}
#'
#' @param dimension An integer representing the dimension of the correlation matrices.
#' @param concentration A numeric vector representing the concentration/shape parameter of the distribution (often referred to as eta). Must be positive.
#' @param sample_method (str): Either "cvine" or "onion". Methods proposed offer the same distribution over correlation matrices. But they are different in how to generate samples. Defaults to "onion".
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
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
#'    - When \code{sample=FALSE}: A BI LKJ distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the LKJ distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).

#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lkj}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.lkj( dimension = 2, concentration=1.0, shape = c(1), sample = TRUE)
#' }
#' @export
bi.dist.lkj=function(dimension, concentration=1.0, sample_method='onion', validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     dimension=as.integer(dimension);
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$lkj(
       dimension = dimension,
       concentration= .BI_env$jnp$array(concentration),
       sample_method= sample_method,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

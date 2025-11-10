#' @title Truncated Normal Distribution
#' @description The Truncated Normal distribution is a normal distribution truncated
#' to a specified interval. It is defined by its location (`loc`), scale
#' (`scale`), lower bound (`low`), and upper bound (`high`).
#'
#' @param loc The location parameter of the normal distribution.
#' @param scale The scale parameter of the normal distribution.
#' @param low (float, jnp.ndarray, optional): The lower truncation point. If `None`, the distribution is only truncated on the right. Defaults to `None`.
#' @param high (float, jnp.ndarray, optional): The upper truncation point. If `None`, the distribution is only truncated on the left. Defaults to `None`.
#' @param shape A numeric vector (e.g., `c(10)`) used to shape the distribution.
#'   When `sample=False` (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the
#'   given shape.
#' @param event The number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution
#'   object instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BI Truncated Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_normal(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bi.dist.truncated_normal=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$truncated_normal(
       loc=.BI_env$jnp$array(loc),
       scale= .BI_env$jnp$array(scale),
       low= .BI_env$jnp$array(low),
       high = .BI_env$jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

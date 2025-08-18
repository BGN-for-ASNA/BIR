#' @title Uniform Distribution
#'
#' @description
#' Samples from a Uniform distribution, which is a continuous probability distribution
#' where all values within a given interval are equally likely.
#'
#' \deqn{f(x) = \frac{1}{b - a}, \text{ for } a \le x \le b}
#'
#' @param low A numeric vector, matrix, or array representing the lower bound of the uniform interval.
#' @param high A numeric vector, matrix, or array representing the upper bound of the uniform interval.
#' @param shape A numeric vector specifying the shape of the output. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions
#'   (used in model building).
#' @param mask A logical vector, matrix, or array (optional) to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BI Uniform distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Uniform distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.uniform(low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bi.dist.uniform=function(low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .BI_env$.bi_instance$dist$uniform(
       low = .BI_env$jnp$array(low),
       high = .BI_env$jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Gamma-Poisson Distribution
#' @description A compound distribution comprising of a gamma-poisson pair, also referred to as
#' a gamma-poisson mixture. The ``rate`` parameter for the
#' :class:`bi.dist.poisson` distribution is unknown and randomly
#' drawn from a :class:`bi.dist.gamma` distribution.
#'
#' @param concentration A numeric vector, matrix, or array representing the shape parameter (alpha) of the Gamma distribution.
#' @param rate A numeric vector, matrix, or array representing the rate parameter (beta) for the Gamma distribution.
#' @param shape A numeric vector used to shape the distribution. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
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
#'
#'  - When \code{sample=FALSE}, a BI Gamma-Poisson distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Gamma-Poisson distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#gammapoisson}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.gamma_poisson(concentration = 1, sample = TRUE)
#' }
#' @export
#'
bi.dist.gamma_poisson=function(concentration, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$gamma_poisson(
       concentration = .BI_env$jnp$array(concentration),
       rate= .BI_env$jnp$array(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

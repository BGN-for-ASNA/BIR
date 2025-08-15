#' @title Samples from a Beta-Proportion distribution.
#'
#' @description The Beta-Proportion  distribution is a reparameterization of the conventional
#' Beta distribution in terms of a the variate mean and a
#' precision parameter. It's useful for modeling rates and proportions.
#'
#' \deqn{f(x) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}}
#'
#' @param mean A numeric vector, matrix, or array representing the mean of the BetaProportion distribution,
#'   must be between 0 and 1.
#' @param concentration A numeric vector, matrix, or array representing the concentration parameter of the BetaProportion distribution.
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
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Beta-Proportion distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Beta-Proportion distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.beta_proportion(0, 1, sample = TRUE)
#' }
#' @export
bi.dist.beta_proportion=function(mean, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$beta_proportion(
       mean = jnp$array(mean),
       concentration  = jnp$array(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

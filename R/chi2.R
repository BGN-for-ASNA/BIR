#' @title Samples from a Chi-squared distribution.
#'
#' @description The Chi-squared distribution is a continuous probability distribution that arises
#' frequently in hypothesis testing, particularly in ANOVA and chi-squared tests.
#' It is defined by a single positive parameter, degrees of freedom (df), which
#' determines the shape of the distribution.
#'
#' \deqn{p(x; df) = \frac{1}{2^{df/2} \Gamma(df/2)} x^{df/2 - 1} e^{-x/2}}

#' @param df A numeric vector representing the degrees of freedom. Must be positive.
#' @param shape A numeric vector used for shaping. When `sample=FALSE` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
#'   of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site. This is essential for building complex distributions
#'   like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Chi-squared distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Chi-squared distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#chi2}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.chi2(c(0,2),sample = TRUE)
#' }
#' @export
#'
bi.dist.chi2=function(df, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$chi2(
       df = jnp$array(df),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

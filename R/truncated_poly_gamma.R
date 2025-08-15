#' @title Truncated PolyaGamma Distribution
#' @description
#' Samples from a Truncated PolyaGamma distribution.
#'
#' This distribution is a truncated version of the PolyaGamma distribution,
#' defined over the interval [0, truncation_point]. It is often used in
#' Bayesian non-parametric models.
#'
#' \deqn{p(x) = \frac{1}{Z} \exp\left( \sum_{n=0}^{N} \left( \log(2n+1) - 1.5 \log(x) - \frac{(2n+1)^2}{4x} \right) \right)}
#'
#' @param batch_shape A numeric vector specifying the shape of the batch dimension.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A numeric vector, matrix, or array (e.g., a JAX array) of boolean values to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Truncated PolyaGamma distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated PolyaGamma distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_polya_gamma(batch_shape = c(), sample = TRUE)
#' }
#' @export
bi.dist.truncated_polya_gamma=function(batch_shape=c(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     batch_shape=do.call(tuple, as.list(as.integer(batch_shape)))
     seed=as.integer(seed);
     .bi$dist$truncated_polya_gamma(
       batch_shape,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

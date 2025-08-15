#' @title Samples from a Normal (Gaussian) distribution.
#' @description
#' The Normal distribution is characterized by its mean (loc) and standard deviation (scale).
#' It's a continuous probability distribution that arises frequently in statistics and
#' probability theory.
#'
#' \deqn{p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
#'
#' @param loc A numeric vector, matrix, or array representing the mean of the distribution.
#' @param scale A numeric vector, matrix, or array representing the standard deviation of the distribution.
#' @param shape A numeric vector specifying the shape of the distribution.  Use a vector (e.g., `c(10)`) to define the shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}: A BI Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}: A JAX array of samples drawn from the Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.normal(sample = TRUE)
#' }
#' @export
bi.dist.normal=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$normal(
       loc=jnp$array(loc),
       scale= jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

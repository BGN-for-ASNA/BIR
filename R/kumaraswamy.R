#' @title  Kumaraswamy Distribution
#'
#' @description The Kumaraswamy distribution is a continuous probability distribution defined on the interval [0, 1].
#' It is a flexible distribution that can take on various shapes depending on its parameters.
#'
#' \deqn{f(x; a, b) = a b x^{a b - 1} (1 - x)^{b - 1}}
#'
#' @export
#' @importFrom reticulate py_none tuple
#'
#' @param concentration1 A numeric vector, matrix, or array representing the first shape parameter. Must be positive.
#' @param concentration0 A numeric vector, matrix, or array representing the second shape parameter. Must be positive.
#' @param shape A numeric vector. When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'    - When \code{sample=FALSE}: A BI Kumaraswamy distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the Kumaraswamy distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#kumaraswamy}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.kumaraswamy(concentration1 = 5, concentration0 = 1., sample = TRUE)
#' }
#' @export

bi.dist.kumaraswamy=function(concentration1, concentration0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$kumaraswamy(
       concentration1 = jnp$array(concentration1),
       concentration0 = jnp$array(concentration0),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

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
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.lkj( dimension = 2, concentration=1.0, shape = c(1), sample = TRUE)
#' }
#' @export
bi.dist.lkj=function(dimension, concentration=1.0, sample_method='onion', validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     dimension=as.integer(dimension);
     seed=as.integer(seed);
     .bi$dist$lkj(
       dimension = dimension,
       concentration= jnp$array(concentration),
       sample_method= sample_method,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

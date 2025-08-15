#' @title A finite mixture of component distributions from the same family.
#' @description
#' This mixture only supports a mixture of component distributions that are all
#' of the same family. The different components are specified along the last
#' batch dimension of the input ``component_distribution``. If you need a
#' mixture of distributions from different families, use the more general
#' implementation in :class:`bi.dist.mixture_general`.
#'
#' @param loc A numeric vector, matrix, or array representing the location parameter of the component distribution.
#' @param scale A numeric vector, matrix, or array representing the scale parameter of the component distribution.
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#' @return
#'    - When \code{sample=FALSE}, a BI MixtureSameFamily distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MixtureSameFamily distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#mixture-same-family}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.mixture_same_family(
#' mixing_distribution = bi.dist.categorical(probs = c(0.3, 0.7),create_obj = TRUE),
#' component_distribution = bi.dist.normal(0,1, shape = c(2), create_obj = TRUE),
#' sample = TRUE)
#' }
#' @export
bi.dist.mixture_same_family=function(mixing_distribution, component_distribution, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$mixture_same_family(
       mixing_distribution,
       component_distribution ,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title A marginalized finite mixture of component distributions.
#' @description
#' This distribution represents a mixture of component distributions, where the
#' mixing weights are determined by a Categorical distribution. The resulting
#' distribution can be either a MixtureGeneral (when component distributions
#' are a list) or a MixtureSameFamily (when component distributions are a single
#' distribution).
#'
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A logical vector used to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object.
#' @return When \code{sample=FALSE}: A BI Mixture distribution object (for model building).
#'         When \code{sample=TRUE}: A JAX array of samples drawn from the Mixture distribution (for direct sampling).
#'         When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#' @seealso
#'  - When \code{sample=FALSE}, a BI marginalized finite mixture distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the marginalized finite mixture distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.mixture(
#'   mixing_distribution = bi.dist.categorical(probs = c(0.3, 0, 7),create_obj = TRUE),
#'   component_distributions = c(
#'   bi.dist.normal(0,1,create_obj = TRUE),
#'   bi.dist.normal(0,1,create_obj = TRUE),
#'   bi.dist.normal(0,1,create_obj = TRUE)
#'   ),
#'   sample = TRUE
#' )
#' }
#' @export

bi.dist.mixture=function(mixing_distribution, component_distributions, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$mixture(
       mixing_distribution,
       component_distributions = reticulate::r_to_py(component_distributions, convert = TRUE),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

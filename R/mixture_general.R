#' @title A finite mixture of component distributions from different families.
#'
#' @description
#' A finite mixture of component distributions from different families.
#'
#' @param mixing_distribution A `Categorical` distribution specifying the weights for each mixture component.
#'   The size of this distribution specifies the number of components in the mixture.
#' @param component_distributions A list of distributions representing the components of the mixture.
#' @param support A constraint object specifying the support of the mixture distribution.
#'   If not provided, the support will be inferred from the component distributions.
#'
#' @return
#'    - When \code{sample=FALSE}, a BI MixtureGeneral distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MixtureGeneral distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).

#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.mixture_general(
#' mixing_distribution = bi.dist.categorical(probs = c(0.3, 0, 7),create_obj = TRUE),
#' component_distributions = c(
#' bi.dist.normal(0,1,create_obj = TRUE),
#' bi.dist.normal(0,1,create_obj = TRUE),
#' bi.dist.normal(0,1,create_obj = TRUE)),
#' sample = TRUE)
#' }
#' @export
bi.dist.mixture_general=function(mixing_distribution, component_distributions, support=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$mixture_general(
       mixing_distribution,
       component_distributions = reticulate::r_to_py(component_distributions, convert = TRUE),
       support= support,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

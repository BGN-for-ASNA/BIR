#' @title Samples from a Dirichlet Multinomial distribution.
#'
#' @description This distribution combines a Dirichlet distribution (for the probabilities of categories)
#' and a Multinomial distribution (for the counts within each category).  The Dirichlet
#' distribution acts as a prior on the probabilities, allowing for a flexible and
#' informative model.
#'
#' @title Dirichlet Multinomial
#' @description Samples from a Dirichlet Multinomial  distribution.
#' @param concentration A numeric vector or array representing the concentration parameter (alpha) for the Dirichlet distribution.
#' @param shape A numeric vector specifying the shape of the distribution. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape. When
#'   `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX
#'   array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Dirichlet Multinomial  distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Dirichlet Multinomial  distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.dirichlet_multinomial(concentration = c(0,1), sample = TRUE, shape = (3))
#' }
#' @export

bi.dist.dirichlet_multinomial=function(concentration, total_count=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$dirichlet_multinomial(
       concentration = jnp$array(concentration),
       total_count = jnp$array(as.integer(total_count)),
       validate_args = validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

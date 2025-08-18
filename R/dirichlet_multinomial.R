#' @title Samples from a Dirichlet Multinomial distribution.
#'
#' @description This distribution combines a Dirichlet distribution (for the probabilities of categories)
#' and a Multinomial distribution (for the counts within each category).  The Dirichlet
#' distribution acts as a prior on the probabilities, allowing for a flexible and
#' informative model.
#'
#' @param concentration A numeric vector or array representing the concentration parameter (alpha) for the Dirichlet distribution.
#' @param total_count (int, jnp.ndarray, optional): The total number of trials (n). This must be a non-negative integer. Defaults to 1.
#' @param shape A numeric vector specifying the shape of the distribution. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape. When
#'   `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX
#'   array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
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
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.dirichlet_multinomial(concentration = c(0,1), sample = TRUE, shape = (3))
#' }
#' @export

bi.dist.dirichlet_multinomial=function(concentration, total_count=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .BI_env$.bi_instance$dist$dirichlet_multinomial(
       concentration = .BI_env$jnp$array(concentration),
       total_count = .BI_env$jnp$array(as.integer(total_count)),
       validate_args = validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

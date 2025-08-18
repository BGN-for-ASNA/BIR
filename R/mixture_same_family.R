#' @title A finite mixture of component distributions from the same family.
#' @description
#' This mixture only supports a mixture of component distributions that are all
#' of the same family. The different components are specified along the last
#' batch dimension of the input ``component_distribution``. If you need a
#' mixture of distributions from different families, use the more general
#' implementation in :class:`bi.dist.mixture_general`.
#'
#' @param mixing_distribution A distribution specifying the weights for each mixture component.
#'   The size of this distribution specifies the number of components in the mixture.
#' @param component_distribution A list of distributions representing the components of the mixture.
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
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

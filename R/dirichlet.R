#' @title Samples from a Dirichlet distribution.
#'
#' @description
#' The Dirichlet distribution is a multivariate generalization of the Beta distribution.
#'It is a probability distribution over a simplex, which is a set of vectors where each element is non-negative and sums to one. It is often used as a prior distribution for categorical distributions.

#' @param concentration A numeric vector or array representing the concentration parameter(s) of the Dirichlet distribution. Must be positive.
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BF distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BF Dirichlet distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Dirichlet distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#dirichlet}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.dirichlet(concentration =  c(0.1,.9), sample = TRUE)
#' }
#' @export
bf.dist.dirichlet=function(concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$dirichlet(
       concentration = .BF_env$jnp$asarray(concentration),
       validate_args = validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

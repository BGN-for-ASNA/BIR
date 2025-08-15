#' @title Samples from a Dirichlet distribution.
#'
#' @description The Dirichlet distribution is a multivariate generalization of the Beta distribution.
#' It is a probability distribution over a simplex, which is a set of vectors where each element is non-negative and sums to one.
#' It is often used as a prior distribution for categorical distributions.
#'
#' \deqn{P(x_1, ..., x_K) = \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i - 1}}
#'
#' @param concentration A numeric vector or array representing the concentration parameter(s) of the Dirichlet distribution. Must be positive.
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Dirichlet distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Dirichlet distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#dirichlet}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.dirichlet(concentration =  c(0.1,.9), sample = TRUE)
#' }
#' @export
bi.dist.dirichlet=function(concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$dirichlet(
       concentration = jnp$array(concentration),
       validate_args = validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

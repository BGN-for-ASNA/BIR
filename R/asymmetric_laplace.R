#' @title Asymmetric Laplace distribution wrapper.
#' @description Samples from an Asymmetric Laplace distribution.
#' The Asymmetric Laplace distribution is a generalization of the Laplace distribution,
#' where the two sides of the distribution are scaled differently. It is defined by
#' a location parameter (`loc`), a scale parameter (`scale`), and an asymmetry parameter (`asymmetry`).
#'
#' The probability density function is:
#' \deqn{f(x; \mu, \sigma, \kappa) = \frac{1}{\sigma(\kappa + 1/\kappa)} \exp\left(-\frac{(x-\mu)\text{sgn}(x-\mu)}{\sigma\kappa^{\text{sgn}(x-\mu)}}\right)}
#' where \eqn{\mu} is the location, \eqn{\sigma} is the scale, and \eqn{\kappa} is the asymmetry.
#' For \eqn{x < \mu}, the scale is \eqn{\sigma\kappa}, and for \eqn{x > \mu}, the scale is \eqn{\sigma/\kappa}.
#'
#' @param loc A numeric vector or single numeric value representing the location parameter of the distribution. This corresponds to \eqn{\mu}.
#' @param scale A numeric vector or single numeric value representing the scale parameter of the distribution. This corresponds to \eqn{\sigma}.
#' @param asymmetry A numeric vector or single numeric value representing the asymmetry parameter of the distribution. This corresponds to \eqn{\kappa}.
#' @param shape A numeric vector specifying the shape of the output.  This is used to set the batch shape when \code{sample=FALSE} (model building) or as `sample_shape` to draw a raw JAX array when \code{sample=TRUE} (direct sampling).
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector indicating which observations to mask.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return When \code{sample=FALSE}: A BI AsymmetricLaplace distribution object (for model building).
#'         When \code{sample=TRUE}: A JAX array of samples drawn from the AsymmetricLaplace distribution (for direct sampling).
#'         When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.asymmetric_laplace(sample = TRUE)
#' }
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplace}
#' @export
bi.dist.asymmetric_laplace=function(loc=0.0, scale=1.0, asymmetry=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$asymmetric_laplace(
       loc=jnp$array(loc),
       scale= jnp$array(scale),
       asymmetry= jnp$array(asymmetry),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

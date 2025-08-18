#' @title Laplace Distribution
#'
#' @description Samples from a Laplace distribution, also known as the double exponential distribution.
#' The Laplace distribution is defined by its location parameter (loc) and scale parameter (scale).
#'
#' \deqn{f(x) = \frac{1}{2s} \exp\left(-\frac{|x - \mu|}{s}\right)}
#'
#' @param loc A numeric vector representing the location parameter of the Laplace distribution.
#' @param scale A numeric vector representing the scale parameter of the Laplace distribution. Must be positive.
#' @param shape A numeric vector used for shaping. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, optionally used to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a
#'   sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'    - When \code{sample=FALSE}: A BI Laplace distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the Laplace distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#laplace}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.laplace(sample = TRUE)
#' }
#' @export
bi.dist.laplace=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$laplace(
       loc = jnp$array(loc),
       scale = jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

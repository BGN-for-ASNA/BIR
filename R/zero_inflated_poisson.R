#' @title A Zero Inflated Poisson distribution.
#' @description
#' This distribution combines two Poisson processes: one with a rate parameter and another that generates only zeros.
#' The probability of observing a zero is determined by the 'gate' parameter, while the probability of observing a non-zero value is governed by the 'rate' parameter of the underlying Poisson distribution.
#'
#' \deqn{P(X = k) = (1 - gate) * \frac{e^{-rate} rate^k}{k!} + gate}
#'
#' @param gate The gate parameter.
#' @param rate A numeric vector, matrix, or array representing the rate parameter of the underlying Poisson distribution.
#' @param shape A numeric vector used to shape the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector, matrix, or array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#' @param sample Logical; If `TRUE`, draws samples from the distribution.
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
#'  #' @return
#'  - When \code{sample=FALSE}, a BI Zero Inflated Poisson distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero Inflated Poisson distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.zero_inflated_poisson(gate=0.3, rate = 5, sample = TRUE)
#' }
#' @export
bi.dist.zero_inflated_poisson=function(gate, rate=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$zero_inflated_poisson(
       gate = .BI_env$jnp$array(gate),
       rate = .BI_env$jnp$array(rate),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

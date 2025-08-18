#' @title Samples from a Gaussian Random Walk distribution.
#'
#' @description A Gaussian Random Walk is a stochastic process where each step is a Gaussian-distributed increment.
#' It can be thought of as a discrete-time version of a Brownian motion.
#'
#' \deqn{X_{t} = \sum_{i=1}^{t} \epsilon_i}
#'
#' where \eqn{\epsilon_i \sim \mathcal{N}(0, \sigma^2)} are independent Gaussian random variables.
#'
#' @param scale A numeric value representing the standard deviation of the Gaussian increments.
#' @param num_steps (int, optional): The number of steps in the random walk, which determines the event shape of the distribution. Must be positive. Defaults to 1.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array. Optional boolean array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BI Gaussian Random Walk distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Gaussian Random Walk distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.gaussian_random_walk(scale = c(1,5,10), sample = TRUE)
#' }
#' @export
bi.dist.gaussian_random_walk=function(scale=1.0, num_steps=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     num_steps=as.integer(num_steps);
     .bi$dist$gaussian_random_walk(
       scale = jnp$array(scale),
       num_steps = num_steps,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

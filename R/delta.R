#' @title The Delta distribution.
#'
#' @description The Delta distribution, also known as a point mass distribution, assigns probability 1 to a single point and 0 elsewhere. It's useful for representing deterministic variables or as a building block for more complex distributions.
#'
#' \deqn{P(x = v) = 1}
#'
#' @importFrom reticulate py_none tuple
#' @param log_density The log probability density of the point mass. This is primarily for creating distributions that are non-normalized or for specific advanced use cases. For a standard delta distribution, this should be 0. Defaults to 0.0.
#' @param v A numeric vector representing the location of the point mass.
#' @param event_dim event_dim (A numeric vector, optional): The number of rightmost dimensions of `v` to interpret as event dimensions. Defaults to 0.
#' @param shape A numeric vector used for shaping. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A boolean vector to mask observations.
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
#'  - When \code{sample=FALSE}, a BI Delta distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Delta distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#delta}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.delta(v = 5, sample = TRUE)
#' }
#' @export
#'
bi.dist.delta=function(v=0.0, log_density=0.0, event_dim=0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event_dim=as.integer(event)
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$delta(
       v=jnp$array(v),
       log_density= jnp$array(log_density),
       event_dim= jnp$array(event_dim),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

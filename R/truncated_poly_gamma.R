#' @title Truncated PolyaGamma Distribution
#' @description
#' This distribution is a truncated version of the PolyaGamma distribution,
#' defined over the interval [0, truncation_point]. It is often used in
#' Bayesian non-parametric models.
#'
#' \deqn{p(x) = \frac{1}{Z} \exp\left( \sum_{n=0}^{N} \left( \log(2n+1) - 1.5 \log(x) - \frac{(2n+1)^2}{4x} \right) \right)}
#'
#' @param batch_shape A numeric vector specifying the shape of the batch dimension.
#' @param shape A numeric vector (e.g., `c(10)`) used to shape the distribution.
#'   When `sample=False` (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the
#'   given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions.
#' @param mask A numeric vector, matrix, or array (e.g., a JAX array) of boolean values to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BF distribution object instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BF Truncated PolyaGamma distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated PolyaGamma distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.truncated_polya_gamma(batch_shape = c(), sample = TRUE)
#' }
#' @export
bf.dist.truncated_polya_gamma=function(batch_shape=c(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     batch_shape=do.call(tuple, as.list(as.integer(batch_shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$truncated_polya_gamma(
       batch_shape,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

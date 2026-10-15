#' @title Samples from a Normal (Gaussian) distribution.
#' @description
#' The Normal distribution is characterized by its mean (loc) and standard deviation (scale).
#' It's a continuous probability distribution that arises frequently in statistics and  probability theory.
#' \deqn{
#' f(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
#' }
#' @param loc A numeric vector, matrix, or array representing the mean of the distribution.
#' @param scale A numeric vector, matrix, or array representing the standard deviation of the distribution.
#' @param shape A numeric vector specifying the shape of the distribution.  Use a vector (e.g., `c(10)`) to define the shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array. Optional boolean array to mask observations.
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
#'  - When \code{sample=FALSE}: A BF Normal distribution object (for model building).
#'
#'  - When \code{sample=TRUE}: A JAX array of samples drawn from the Normal distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}: The raw BF distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.normal(sample = TRUE)
#' }
#' @export
bf.dist.normal=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$normal(
       loc=.BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,  to_jax = to_jax)
}

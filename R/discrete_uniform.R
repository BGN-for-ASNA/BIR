#' @title Samples from a Discrete Uniform distribution.
#'
#' @description
#' The Discrete Uniform distribution defines a uniform distribution over a range of integers.
#' It is characterized by a lower bound (`low`) and an upper bound (`high`), inclusive.
#'
#'  \deqn{
#'   P(X = k) = \frac{1}{high - low + 1}, \quad k \in \{low, low+1, ..., high\}
#' }
#'   Otherwise (if \deqn{k} is outside the range), $ P(X = k) = 0 $.
#'
#' @title DiscreteUniform
#' @description Samples from a Discrete Uniform distribution.
#' @param low A numeric vector representing the lower bound of the uniform range, inclusive.
#' @param high A numeric vector representing the upper bound of the uniform range, inclusive.
#' @param shape A numeric vector.  When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
#' @param infer Optional named list passed to NumPyro's `infer` argument
#'   (e.g. `list(enumerate = "parallel")`).
#' @param enum If `TRUE`, the discrete latent is summed out exactly (parallel
#'   enumeration) instead of sampled; same as `infer = list(enumerate = "parallel")`.
#'   Defaults to `FALSE`.
#' @param create_obj Logical. If TRUE, returns the raw BF distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BF Discrete Uniform distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Discrete Uniform distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.discrete_uniform(sample = TRUE)
#' }
#' @export
bf.dist.discrete_uniform=function(low=0, high=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$discrete_uniform(
       low = .BF_env$jnp$asarray(low),
       high = .BF_env$jnp$asarray(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

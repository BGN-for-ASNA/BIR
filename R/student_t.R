#' @title Student's t-distribution.
#' @description
#' The Student's t-distribution is a probability distribution that arises in hypothesis
#' testing involving the mean of a normally distributed population when the population
#'  standard deviation is unknown. It is similar to the normal distribution, but has heavier tails,
#'   making it more robust to outliers. For large \deqn{ \nu }, it converges to the Normal distribution.
#' \deqn{
#'   X \sim t_\nu(\mu, \sigma)
#' }
#' where:
#'
#' * \deqn{ \mu } is the **location (mean)** parameter
#' * \deqn{ \sigma > 0 } is the **scale** parameter
#' * \deqn{ \nu > 0 } is the **degrees of freedom** controlling the tail heaviness
#'
#' @param df A numeric vector representing degrees of freedom, must be positive.
#' @param loc A numeric vector representing the location parameter, defaults to 0.0.
#' @param scale A numeric vector representing the scale parameter, defaults to 1.0.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BF distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BF Student's t-distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Student's t-distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#studentt}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.student_t(df = 2, loc = 0, scale = 2, sample = TRUE)
#' }
#' @export
bf.dist.student_t=function(df, loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$student_t(
       df = .BF_env$jnp$asarray((df)),
       loc= .BF_env$jnp$asarray(loc),
       scale= .BF_env$jnp$asarray(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Samples from a Negative Binomial distribution.
#' @description
#' The NB2 parameterisation of the negative-binomial distribution is a count distribution used for modelling over-dispersed count data
#' (variance > mean). It is defined such that the variance grows **quadratically** in the mean:

#' \deqn{
#'   Var(Y) = \mu + \alpha,\mu^2,
#' }
#' where \deqn{\mu = E[Y]) and (\alpha>0)} is the dispersion (heterogeneity) parameter.
#' Because of this quadratic variance growth, it is called the NB2 family.
#'
#'
#' \deqn{P(k) = \frac{\Gamma(k + \alpha)}{\Gamma(k + 1) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^k \left(1 - \frac{\beta}{\alpha + \beta}\right)^k}
#'
#' @param total_count (int): The number of trials *n*.
#' @param probs A numeric vector, matrix, or array representing the probability of success for each Bernoulli trial. Must be between 0 and 1.
#' @param logits A numeric vector, matrix, or array representing the log-odds of success for each trial.
#' @param shape A numeric vector.  Used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional logical vector to mask observations.
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
#'  - When \code{sample=FALSE}, a BF Negative Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Negative Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#negativebinomial2}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.negative_binomial2(total_count = 100, probs = 0.5, sample = TRUE)
#' }
#' @export
bf.dist.negative_binomial2=function(total_count, probs, logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if (.BF_env$.py$is_none(logits)){
       .BF_env$.bf_instance$dist$negative_binomial(total_count = .BF_env$jnp$asarray(total_count), probs=.BF_env$jnp$asarray(probs),   validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
     }else{
       .BF_env$.bf_instance$dist$negative_binomial(total_count = .BF_env$jnp$asarray(total_count), logits= .BF_env$jnp$asarray(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)

     }
}

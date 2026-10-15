#' @title Generic Zero Inflated distribution.
#' @description
#' The Zero-Inflated Poisson distribution is a discrete count-distribution designed for data with *more zeros*
#' than would be expected under a standard Poisson. Essentially, it assumes two underlying processes:

#' * With probability \deqn{\pi} you are in a "structural zero" state (i.e., you automatically get a zero count).
#' * With probability \deqn{1 - \pi} you draw from a standard Poisson distribution with parameter \deqn{\lambda}.
#'
#' This results in a mixture distribution that places more mass at zero than a Poisson alone would.
#' It's widely used in, for instance, ecology (species counts with many zeros), insurance/claims problems,
#' and any count-data setting with excess zeros.
#'
#' @param base_dist Distribution: The base distribution to be zero-inflated (e.g., Poisson, NegativeBinomial).
#' @param gate numeric(1): Probability of extra zeros (between 0 and 1).
#' @param gate_logits numeric(1): Log-odds of extra zeros.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param shape numeric(1): A multi-purpose argument for shaping. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.  Provide as a numeric vector (e.g., `c(10)`).
#' @param event int(1): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask logical(1): Optional boolean array to mask observations.
#' @param create_obj Logical: If True, returns the raw BF distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
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
#'
#'  - When \code{sample=FALSE}, a BF Zero Inflated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero Inflated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution}
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m=importBF(platform='cpu')
#' bf.dist.zero_inflated_distribution(
#' base_dist = bf.dist.poisson(5, create_obj = TRUE),
#' gate=0.3, sample = TRUE)
#' }
#' @export
bf.dist.zero_inflated_distribution=function(base_dist, gate=py_none(), gate_logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if(!.BF_env$.py$is_none(gate)){gate = .BF_env$jnp$asarray(gate)}
     if(!.BF_env$.py$is_none(gate_logits)){gate_logits = .BF_env$jnp$asarray(gate_logits)}

     .BF_env$.bf_instance$dist$zero_inflated_distribution(
       base_dist,
       gate =  .BF_env$jnp$asarray(gate),
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Zero-Inflated Negative Binomial  Distribution
#' @description
#' A Zero-Inflated Negative Binomial distribution is used for count data that exhibit **both** (a)
#' over-dispersion relative to a Poisson (i.e., variance > mean) *and* (b)
#' an excess of zero counts beyond what a standard Negative Binomial would predict.
#' It assumes two latent processes:

#'  1. With probability \deqn{\pi } (sometimes denoted \deqn{\psi} or "zero-inflation probability")
#'  you are in a "structural zero" state ??? you observe a zero.
#'  2. With probability \deqn{1 - \pi}, you come from a regular Negative Binomial distribution
#'  (with parameters e.g. mean \deqn{\mu} and dispersion parameter \deqn{ \alpha }
#'   or size/r parameter) and then you might observe zero or a positive count.

#'  Thus the model is a mixture of a point-mass at zero + a Negative Binomial for counts.

#'  This distribution combines a Negative Binomial distribution with a binary gate variable. Observations are
#'  either drawn from the Negative Binomial distribution with probability (1 - gate) or are treated as zero with probability 'gate'.
#'
#'   This models data with excess zeros compared to what a standard Negative Binomial distribution would predict.
#' @param mean Numeric or a numeric vector. The mean of the Negative Binomial 2 distribution.
#' @param concentration Numeric or a numeric vector. The concentration parameter of the Negative Binomial 2 distribution.
#' @param gate numeric(1): Probability of extra zeros (between 0 and 1).
#' @param gate_logits numeric(1): Log-odds of extra zeros.
#' @param shape A numeric vector.  A multi-purpose argument for shaping. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw NumPyro distribution object instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BF Zero-Inflated Negative Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero-Inflated Negative Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BF distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.zero_inflated_negative_binomial(mean = 2, concentration = 1, gate = 0.3, sample = TRUE)
#' }
#' @export
bf.dist.zero_inflated_negative_binomial=function(mean, concentration, gate=py_none(), gate_logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if(!.BF_env$.py$is_none(gate)){gate = .BF_env$jnp$asarray(gate)}
     if(!.BF_env$.py$is_none(gate_logits)){gate_logits = .BF_env$jnp$asarray(gate_logits)}


     .BF_env$.bf_instance$dist$zero_inflated_negative_binomial2(
       mean = .BF_env$jnp$asarray(mean),
       concentration = .BF_env$jnp$asarray(concentration),
       gate = gate,
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

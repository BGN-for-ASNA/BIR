#' @title Generic Zero Inflated distribution.
#' @description
#' A Zero-Inflated distribution combines a base distribution with a Bernoulli
#' distribution to model data with an excess of zero values. It assumes that each observation
#' is either drawn from the base distribution or is a zero with probability determined
#' by the Bernoulli distribution (the "gate"). This is useful for modeling data
#' where zeros are more frequent than expected under a single distribution,
#' often due to a different underlying process.
#'
#' \deqn{P(x) = \pi \cdot I(x=0) + (1 - \pi) \cdot P_{base}(x)}
#'
#' where:
#'
#' - \eqn{P_{base}(x)} is the probability density function (PDF) or probability mass function (PMF) of the base distribution.
#'
#' - \eqn{\pi} is the probability of generating a zero, governed by the Bernoulli gate.
#'
#' - \eqn{I(x=0)} is an indicator function that equals 1 if x=0 and 0 otherwise.
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
#' @param create_obj Logical: If True, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'
#'  - When \code{sample=FALSE}, a BI Zero Inflated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero Inflated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.zero_inflated_distribution(base_dist = bi.dist.poisson(5, create_obj = TRUE), gate=0.3, sample = TRUE)
#' }
#' @export
bi.dist.zero_inflated_distribution=function(base_dist, gate=py_none(), gate_logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);

     if(!py$is_none(gate)){gate = jnp$array(gate)}
     if(!py$is_none(gate_logits)){gate_logits = jnp$array(gate_logits)}

     .bi$dist$zero_inflated_distribution(
       base_dist,
       gate =  jnp$array(gate),
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

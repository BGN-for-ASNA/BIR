#' @title Zero-Inflated Negative Binomial Distribution
#' @description
#' This distribution combines a Negative Binomial distribution with a binary gate variable.
#' Observations are either drawn from the Negative Binomial distribution with probability
#' (1 - gate) or are treated as zero with probability 'gate'. This models data with excess zeros
#' compared to what a standard Negative Binomial 2 distribution would predict.
#'
#' \deqn{P(X = x) = (1 - gate) \cdot \frac{\Gamma(x + \alpha)}{\Gamma(x + \alpha + \beta) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^x + gate \cdot \delta_{x, 0}}
#'
#'
#' @param mean Numeric or a numeric vector. The mean of the Negative Binomial distribution.
#' @param concentration Numeric or a numeric vector. The concentration parameter of the Negative Binomial 2 distribution.
#' @param shape A numeric vector.  A multi-purpose argument for shaping. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Zero-Inflated Negative Binomial distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Zero-Inflated Negative Binomial distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflatednegativebinomial2}
#'
#' @examples
#' #' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.zero_inflated_negative_binomial(
#'   mean = 2,
#'   concentration = 1,
#'   gate=0.3,
#'   sample = TRUE
#' )
#' }
#' @export
bi.dist.zero_inflated_negative_binomial=function(
    mean, concentration, gate=py_none(),
    gate_logits=py_none(), validate_args=py_none(),
    name='x', obs=py_none(), mask=py_none(),
    sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);

     if(!py$is_none(gate)){gate = jnp$array(gate)}
     if(!py$is_none(gate_logits)){gate_logits = jnp$array(gate_logits)}


     .bi$dist$zero_inflated_negative_binomial2(
       mean = jnp$array(mean),
       concentration = jnp$array(concentration),
       gate = gate,
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

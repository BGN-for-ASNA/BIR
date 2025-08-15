#' @title Samples from a Relaxed Bernoulli distribution.
#' @description
#' The Relaxed Bernoulli distribution is a continuous relaxation of the discrete Bernoulli distribution.
#' It's useful for variational inference and other applications where a differentiable approximation of the Bernoulli is needed.
#' The probability density function (PDF) is defined as:
#' \deqn{p(x) = \frac{1}{2} \left( 1 + \tanh\left(\frac{x - \beta \log(\frac{p}{1-p})}{1}\right) \right)}
#'
#' @param temperature A numeric value representing the temperature parameter.
#' @param shape A numeric vector (e.g., `c(10)`) specifying the shape. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Relaxed Bernoulli distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Relaxed Bernoulli distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli}
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.relaxed_bernoulli(temperature = c(1,1), logits = 0.0, sample = TRUE)
#' }
#' @export
bi.dist.relaxed_bernoulli=function(temperature, probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     if(!py$is_none(logits)){logits=jnp$array(logits)}
     if(!py$is_none(probs)){probs=jnp$array(probs)}
     .bi$dist$relaxed_bernoulli(
       temperature = jnp$array(temperature),
       probs = probs,
       logits = logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Samples from a Relaxed Bernoulli distribution.
#' @description
#' The Relaxed Bernoulli distribution is a continuous relaxation of the discrete Bernoulli distribution.
#' It's useful for variational inference and other applications where a differentiable approximation of the Bernoulli is needed.
#' The probability density function (PDF) is defined as:
#' \deqn{p(x) = \frac{1}{2} \left( 1 + \tanh\left(\frac{x - \beta \log(\frac{p}{1-p})}{1}\right) \right)}
#'
#' @param temperature A numeric value representing the temperature parameter.
#' @param probs (jnp.ndarray, optional): The probability of success. Must be in the interval `[0, 1]`. Only one of `probs` or `logits` can be specified.
#' @param logits A numeric vector or matrix representing the logits parameter.
#' @param shape A numeric vector (e.g., `c(10)`) specifying the shape. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector or array to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'   This is essential for building complex distributions like `MixtureSameFamily`.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
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
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.relaxed_bernoulli(temperature = c(1,1), logits = 0.0, sample = TRUE)
#' }
#' @export
bi.dist.relaxed_bernoulli=function(temperature, probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BI_env$.py$is_none(logits)){logits=.BI_env$jnp$array(logits)}
     if(!.BI_env$.py$is_none(probs)){probs=.BI_env$jnp$array(probs)}
     .BI_env$.bi_instance$dist$relaxed_bernoulli(
       temperature = .BI_env$jnp$array(temperature),
       probs = probs,
       logits = logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

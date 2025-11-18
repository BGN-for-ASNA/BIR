#' @title Samples from a Logistic distribution.
#' @description
#'The Logistic distribution is a continuous probability distribution defined by two parameters: location and scale. It is often used to model growth processes and is closely related to the normal distribution.Its CDF is the logistic (sigmoid) function, which makes it appealing in modeling probabilities, logistic regression, and various growth models. It resembles the normal distribution in shape (bell-shaped, symmetric) but has **heavier tails** (i.e. more probability in the extremes) and simpler closed-form expressions for the CDF.#' @param loc Numeric vector or single number. The location parameter, specifying the median of the distribution. Defaults to 0.0.
#' @param scale Numeric vector or single number. The scale parameter, which determines the spread of the distribution. Must be positive. Defaults to 1.0.
#' @param shape Numeric vector. A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask Logical vector. Optional boolean array to mask observations.
#' @param create_obj Logical. If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BI Logistic distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Logistic distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.logistic(sample = TRUE)
#' }
#' @export
bi.dist.logistic=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$logistic(
       loc = .BI_env$jnp$array(loc),
       scale = .BI_env$jnp$array(scale),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

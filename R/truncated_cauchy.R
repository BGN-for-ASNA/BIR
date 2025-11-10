#' @title Truncated Cauchy Distribution
#' @description
#' Samples from a Truncated Cauchy distribution.
#'
#' The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
#' that appears frequently in various areas of mathematics and physics. It is characterized by its heavy tails,
#' which extend to infinity. The truncated version limits the support of the Cauchy distribution to a specified interval.
#'
#' \deqn{f_{\text{trunc}}(x) = \frac{f_{\text{base}}(x)}{F_{\text{base}}(\text{high}) - F_{\text{base}}(\text{low})} \quad \text{for } \text{low} \le x \le \text{high}}
#'
#' @param loc Location parameter of the Cauchy distribution.
#' @param scale Scale parameter of the Cauchy distribution.
#' @param low (float, jnp.ndarray, optional): The lower truncation point. If `None`, the distribution is only truncated on the right. Defaults to `None`.
#' @param high (float, jnp.ndarray, optional): The upper truncation point. If `None`, the distribution is only truncated on the left. Defaults to `None`.
#' validate_args (bool, optional): Whether to enable validation of distribution parameters. Defaults to `None`.
#' @param shape A numeric vector. When `sample=False` (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#'  - When \code{sample=FALSE}, a BI Truncated Cauchy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated Cauchy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_cauchy(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
#' }
#' @export
bi.dist.truncated_cauchy=function(loc=0.0, scale=1.0, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$truncated_cauchy(
       loc=.BI_env$jnp$array(loc),
       scale= .BI_env$jnp$array(scale),
       low= .BI_env$jnp$array(low),
       high = .BI_env$jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

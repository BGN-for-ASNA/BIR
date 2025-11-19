#' @title Lower Truncated Power Law Distribution
#'
#' @description
#' The *Lower-Truncated Power-Law* distribution (also known as the *Pareto Type I* or *power-law with a lower bound*) models quantities that follow a heavy-tailed power-law behavior but are bounded below by a minimum value ( x_{\min} ). It is commonly used to describe phenomena such as wealth distributions, city sizes, and biological scaling laws.

#' Lower truncated power law distribution with $\alpha` index.

#' @param alpha A numeric vector: index of the power law distribution. Must be less than -1.
#' @param low A numeric vector: lower bound of the distribution. Must be greater than 0.
#' @param shape A numeric vector: A multi-purpose argument for shaping. When `sample=False` (model building), this is used
#'   with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer: The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector: Optional boolean array to mask observations.
#' @param create_obj Logical: If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Lower Truncated Power Law distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Lower Truncated Power Law distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lowertruncatedpowerlaw}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.lower_truncated_power_law( alpha = c(-2, 2), low = c(1, 0.5),  sample = TRUE)
#' }
#' @export

bi.dist.lower_truncated_power_law=function(alpha, low, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$lower_truncated_power_law(
       alpha = .BI_env$jnp$array(alpha),
       low = .BI_env$jnp$array(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

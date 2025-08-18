#' @title Truncated Distribution
#'
#' @description
#' Samples from a Truncated Distribution.
#' This distribution represents a base distribution truncated between specified lower and upper bounds.
#' The truncation modifies the probability density function (PDF) of the base distribution,
#' effectively removing observations outside the defined interval.
#'
#' \deqn{p(x) = \frac{p(x)}{P(\text{lower} \le x \le \text{upper})}}
#'
#' @param base_dist The base distribution to be truncated. This should be a univariate
#'   distribution. Currently, only the following distributions are supported:
#'   Cauchy, Laplace, Logistic, Normal, and StudentT.
#' @param low (float, jnp.ndarray, optional): The lower truncation point. If `None`, the distribution is only truncated on the right. Defaults to `None`.
#' @param high (float, jnp.ndarray, optional): The upper truncation point. If `None`, the distribution is only truncated on the left. Defaults to `None`.
#' @param shape A numeric vector (e.g., `c(10)`) specifying the shape. When \code{sample=FALSE}
#'   (model building), this is used with `.expand(shape)` to set the distribution's
#'   batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape`
#'   to draw a raw JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions
#'   (used in model building).
#' @param mask An optional boolean array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object
#'   instead of creating a sample site. This is essential for building complex distributions
#'   like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BI Truncated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Truncated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#truncateddistribution}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.truncated_distribution(
#' base_dist = bi.dist.normal(0,1, create_obj = TRUE),
#' high = 0.7,
#' low = 0.1,
#' sample = TRUE)
#' }
#' @export
bi.dist.truncated_distribution=function(base_dist, low=py_none(), high=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .BI_env$.bi_instance$dist$truncated_distribution(
       base_dist,
       low = .BI_env$jnp$array(low),
       high = .BI_env$jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title Two-Sided Truncated Distribution
#' @description
#' This distribution truncates a base distribution between two specified lower and upper bounds.
#'
#' \deqn{f(x) = \begin{cases}
#'         \frac{p(x)}{P(\text{low} \le X \le \text{high})} & \text{if } \text{low} \le x \le \text{high} \\
#'         0 & \text{otherwise}
#'     \end{cases}}
#'
#' where \deqn{p(x)} is the probability density function of the base distribution.
#'
#' @param base_dist The base distribution to truncate.
#' @param low The lower bound for truncation.
#' @param high The upper bound for truncation.
#' @param sample Logical; if `TRUE`, returns JAX array of samples.  Defaults to `FALSE`.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector to mask observations.
#' @param create_obj Logical; if `TRUE`, returns the raw BI distribution object. Defaults to `FALSE`.
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
#'  - When \code{sample=FALSE}, a BI Two-Sided Truncated distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Two-Sided Truncated distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.two_sided_truncated_distribution(
#' base_dist = bi.dist.normal(0,1, create_obj = TRUE),
#' high = 0.5, low = 0.1, sample = TRUE)
#' }
#' @export
bi.dist.two_sided_truncated_distribution=function(base_dist, low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$two_sided_truncated_distribution(
       base_dist,
       low= .BI_env$jnp$array(low),
       high = .BI_env$jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

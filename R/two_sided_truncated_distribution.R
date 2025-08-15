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
#' @param create_obj Logical; if `TRUE`, returns the raw BI distribution object. Defaults to `FALSE`.
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
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.two_sided_truncated_distribution(base_dist = bi.dist.normal(0,1, create_obj = TRUE), high = 0.5, low = 0.1, sample = TRUE)
#' }
#' @export
bi.dist.two_sided_truncated_distribution=function(base_dist, low=0.0, high=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$two_sided_truncated_distribution(
       base_dist,
       low= jnp$array(low),
       high = jnp$array(high),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

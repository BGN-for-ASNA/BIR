#' @title Samples from a left-truncated distribution.
#'
#' @description A left-truncated distribution is a probability distribution
#' obtained by restricting the support of another distribution
#' to values greater than a specified lower bound. This is useful
#' when dealing with data that is known to be greater than a certain value.
#'
#'
#' \deqn{f(x) = \begin{cases}
#'           \frac{f(x)}{P(X > \text{low})} & \text{if } x > \text{low} \\
#'           0 & \text{otherwise}
#'           \end{cases}}
#'
#' @param base_dist The base distribution to truncate. Must be univariate and have real support.
#' @param low The lower truncation bound. Values less than this are excluded from the distribution.
#' @param shape A numeric vector. When \code{sample=FALSE} (model building),
#'   this is used with `.expand(shape)` to set the distribution's batch shape.
#'   When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw
#'   JAX array of the given shape.
#' @param event The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#'    - When \code{sample=FALSE}: A BI LeftTruncatedDistribution distribution object (for model building).
#'
#'    - When \code{sample=TRUE}: A JAX array of samples drawn from the LeftTruncatedDistribution distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}: The raw BI distribution object (for advanced use cases).
#'
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#lefttruncateddistribution}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.left_truncated_distribution(
#' base_dist = bi.dist.normal(loc = 1, scale = 10 ,  create_obj = TRUE),
#' sample = TRUE)
#' }
#' @export

bi.dist.left_truncated_distribution=function(base_dist, low=0.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$left_truncated_distribution(
       base_dist,
       low = .BI_env$jnp$array(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Multivariate Student's t Distribution
#'
#' @description
#' The Multivariate Student's t distribution is a generalization of the Student's t
#' distribution to multiple dimensions. It is a heavy-tailed distribution that is
#' often used to model data that is not normally distributed.
#'
#' \deqn{p(x) = \frac{1}{B(df/2, n/2)} \frac{\Gamma(df/2 + n/2)}{\Gamma(df/2)}
#' \left(1 + \frac{(x - \mu)^T \Sigma^{-1} (x - \mu)}{df}\right)^{-(df + n)/2}}
#'
#' @param df A numeric vector representing degrees of freedom, must be positive.
#' @param loc A numeric vector representing the location vector (mean) of the distribution.
#' @param scale_tril A numeric matrix defining the scale (lower triangular matrix).
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector. Optional boolean array to mask observations.
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
#'  - When \code{sample=FALSE}, a BI Multivariate Student's t distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Multivariate Student's t distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multivariatestudentt}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.multivariate_student_t(
#' df = 2,
#' loc =  c(1.0, 0.0, -2.0),
#' scale_tril = chol(
#' matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5),
#' nrow = 3, byrow = TRUE)),
#' sample = TRUE)
#' }
#' @export
bi.dist.multivariate_student_t=function(df, loc=0.0, scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .BI_env$.bi_instance$dist$multivariate_student_t(
       df = .BI_env$jnp$array(df),
       loc = .BI_env$jnp$array(loc),
       scale_tril = .BI_env$jnp$array(scale_tril),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

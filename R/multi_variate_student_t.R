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
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.multivariate_student_t(
#' df = 2,
#' loc =  c(1.0, 0.0, -2.0),
#' scale_tril = jnp$linalg$cholesky(
#' matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5),
#' nrow = 3, byrow = TRUE)),
#' sample = TRUE)
#' }
#' @export
bi.dist.multivariate_student_t=function(df, loc=0.0, scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$multivariate_student_t(
       df = jnp$array(df),
       loc = jnp$array(loc),
       scale_tril = jnp$array(scale_tril),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

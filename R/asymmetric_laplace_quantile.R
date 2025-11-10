#' @title Asymmetric Laplace Quantile Distribution
#'
#' @description Samples from an Asymmetric Laplace Quantile distribution.
#'
#' This distribution is an alternative parameterization of the Asymmetric Laplace
#' distribution, commonly used in Bayesian quantile regression. It utilizes a
#' `quantile` parameter to define the balance between the left- and right-hand
#' sides of the distribution, representing the proportion of probability density
#' that falls to the left-hand side.
#'
#' \deqn{f(x) = \frac{1}{2 \sigma} \exp\left(-\frac{|x - \mu|}{\sigma} \frac{1}{q-1}\right) \left(1 - \frac{1}{2q}\right)}
#'
#' @param loc The location parameter of the distribution.
#'
#' @param scale The scale parameter of the distribution.
#'
#' @param quantile The quantile parameter, representing the proportion of
#'   probability density to the left of the median. Must be between 0 and 1.
#'
#' @param shape A numeric vector. When `sample=False` (model building), this is
#'   used with `.expand(shape)` to set the distribution's batch shape. When
#'   `sample=True` (direct sampling), this is used as `sample_shape` to draw a
#'   raw JAX array of the given shape.
#'
#' @param event The number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#'
#' @param mask An optional boolean array to mask observations.
#'
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#'
#' @param create_obj If `TRUE`, returns the raw NumPyro distribution object
#'   instead of creating a sample site. This is essential for building complex
#'   distributions like `MixtureSameFamily`.
#'
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#'
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#'
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#'
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Asymmetric Laplace Quantile distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Asymmetric Laplace Quantile distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).


#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplacequantile}


#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.asymmetric_laplace_quantile(sample = TRUE)
#' }
#' @export
bi.dist.asymmetric_laplace_quantile=function(loc=0.0, scale=1.0, quantile=0.5, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$asymmetric_laplace_quantile(
       loc=.BI_env$jnp$array(loc),
       scale= .BI_env$jnp$array(scale),
       quantile= .BI_env$jnp$array(quantile),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

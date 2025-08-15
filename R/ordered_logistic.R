#' @title Ordered Logistic Distribution
#' @description
#' A categorical distribution with ordered outcomes. This distribution represents the probability of an event falling into one of several ordered categories, based on a predictor variable and a set of cutpoints. The probability of an event falling into a particular category is determined by the number of categories above it.
#'
#' \deqn{P(Y = k) = \begin{cases}
#'                1 & \text{if } k = 0 \\
#'                \frac{1}{k} & \text{if } k > 0
#'            \end{cases}}
#'
#' @param predictor A numeric vector, matrix, or array representing the prediction in real domain; typically this is output of a linear model.
#' @param cutpoints A numeric vector, matrix, or array representing the positions in real domain to separate categories.
#' @param shape A numeric vector used to shape the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Ordered Logistic distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Ordered Logistic distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#orderedlogistic}
#'
#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.ordered_logistic(predictor = c(0.2, 0.5, 0.8), cutpoints = c(-1.0, 0.0, 1.0), sample = TRUE)
#' }
#' @export
bi.dist.ordered_logistic=function(predictor, cutpoints, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$ordered_logistic(
       predictor = jnp$array(predictor),
       cutpoints = jnp$array(cutpoints),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

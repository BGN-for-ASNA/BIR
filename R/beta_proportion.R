#' @title Samples from a Beta-Proportion distribution.
#'
#' @description
#'The Beta Proportion distribution is a reparameterization of the conventional
#'Beta distribution in terms of a the variate mean and a
#'precision parameter. It's useful for modeling rates and proportions. It's essentially the same family as the standard Beta \deqn{(\alpha,\beta)}, but the mapping is:
#'
#' \deqn{
#'        \alpha = \mu , \kappa,\quad \beta = (1 - \mu), \kappa.
#' }
#' @param mean A numeric vector, matrix, or array representing the mean of the BetaProportion distribution,
#'   must be between 0 and 1.
#' @param concentration A numeric vector, matrix, or array representing the concentration parameter of the BetaProportion distribution.
#' @param shape A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the
#'   distribution's batch shape. When `sample=True` (direct sampling),
#'   this is used as `sample_shape` to draw a raw JAX array of the
#'   given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event
#'   dimensions (used in model building).
#' @param mask An optional boolean vector to mask observations.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution
#'   object instead of creating a sample site. This is essential for
#'   building complex distributions like `MixtureSameFamily`.
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
#'  - When \code{sample=FALSE}, a BI Beta-Proportion distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Beta-Proportion distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.beta_proportion(0, 1, sample = TRUE)
#' }
#' @export
bi.dist.beta_proportion=function(mean, concentration, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$beta_proportion(
       mean = .BI_env$jnp$array(mean),
       concentration  = .BI_env$jnp$array(concentration),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

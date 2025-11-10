#' @title Gaussian State Space Distribution
#'
#' @description Samples from a Gaussian state space model.
#'
#' \deqn{z_{t} = A z_{t - 1} + \epsilon_t \\ z_{t} = \sum_{k=1}^{t} A^{t-k} \epsilon_k}
#'
#' where \eqn{z_t} is the state vector at step \eqn{t}, \eqn{A}
#' is the transition matrix, and \eqn{\epsilon} is the innovation noise.
#'
#' @param num_steps An integer representing the number of steps.
#' @param transition_matrix A numeric vector, matrix, or array representing the state space transition matrix \eqn{A}.
#' @param covariance_matrix A numeric vector, matrix, or array representing the covariance of the innovation noise \eqn{\epsilon}.  Defaults to `reticulate::py_none()`.
#' @param precision_matrix A numeric vector, matrix, or array representing the precision matrix of the innovation noise \eqn{\epsilon}. Defaults to `reticulate::py_none()`.
#' @param scale_tril A numeric vector, matrix, or array representing the scale matrix of the innovation noise \eqn{\epsilon}. Defaults to `reticulate::py_none()`.
#' @param shape A numeric vector specifying the shape. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array representing an optional boolean array to mask observations. Defaults to `reticulate::py_none()`.
#' @param create_obj A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. Defaults to `FALSE`.
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
#' @return When `sample=FALSE`:
#'  - When `sample=FALSE`, a BI  Gaussian State Space distribution object (for model building).
#'
#'  - When `sample=TRUE`, a JAX array of samples drawn from the  Gaussian State Space distribution (for direct sampling).
#'
#'  - When `create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#gaussianstatespace}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.gaussian_state_space(
#'   num_steps = 1,
#'   transition_matrix = matrix(c(0.5), nrow = 1, byrow = TRUE),
#'   covariance_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
#'   sample = TRUE)
#'}
#' @export
bi.dist.gaussian_state_space=function(num_steps, transition_matrix, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     num_steps=as.integer(num_steps);
     if(!.BI_env$.py$is_none(transition_matrix)){transition_matrix = .BI_env$jnp$array(transition_matrix)}
     if(!.BI_env$.py$is_none(covariance_matrix)){covariance_matrix = .BI_env$jnp$array(covariance_matrix)}
     if(!.BI_env$.py$is_none(precision_matrix)){precision_matrix = .BI_env$jnp$array(precision_matrix)}
     if(!.BI_env$.py$is_none(scale_tril)){scale_tril = .BI_env$jnp$array(scale_tril)}


      .BI_env$.bi_instance$dist$gaussian_state_space(
        num_steps,
        transition_matrix = transition_matrix,
        covariance_matrix= covariance_matrix,
        precision_matrix= precision_matrix,
        scale_tril= scale_tril,
        validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

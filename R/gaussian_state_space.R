#' @title Gaussian State Space distribution
#'
#' @description
#' Samples from a Gaussian state space model.
#'
#' @param num_steps Number of steps.
#' @param transition_matrix State space transition matrix \eqn{\mathbf{A}}.
#' @param covariance_matrix Covariance of the innovation noise \eqn{{\epsilon}}.
#'   Defaults to py_none(). - *precision_matrix (numeric vector or array,
#'   optional): Precision matrix of the innovation noise  \eqn{\epsilon}. Defaults
#'   to py_none(). - *scale_tril (numeric vector or array, optional): Scale matrix
#'   of the innovation noise \eqn{\epsilon}. Defaults to py_none().
#' @param precision_matrix See the Python documentation of `precision_matrix`.
#' @param scale_tril See the Python documentation of `scale_tril`.
#' @param validate_args Logical. Whether to validate the parameter values.
#'   Defaults to `py_none()` (the NumPyro default).
#' @param name Character. Name of the random variable in the model. When not
#'   given, an observed site is named after the variable passed to `obs` (`obs =
#'   y` gives `"y"`); use [bf_model()] to name parameters after the variables they
#'   are assigned to.
#' @param obs Numeric vector or array of observed values. If given, the random
#'   variable is conditioned on them (likelihood); otherwise it is latent.
#'   Defaults to `py_none()`.
#' @param mask Optional logical vector or array masking observations (`FALSE` =
#'   ignored). Defaults to `py_none()`.
#' @param sample Logical. If `TRUE`, directly draw values from the distribution
#'   and return them; if `FALSE` (default), create a random variable inside a
#'   model.
#' @param seed Integer seed for this draw when `sample = TRUE`. By default
#'   (`py_none()`) the draw comes from the session's random stream, set by
#'   `importBF(rand_seed = ...)`, so consecutive draws differ but the whole
#'   sequence is reproducible. No effect when `sample = FALSE`.
#' @param shape Integer vector. With `sample = FALSE`, the batch shape of the
#'   distribution in the model; with `sample = TRUE`, the shape of the draw.
#'   Defaults to `c()` (no extra dimensions).
#' @param event Integer. Number of batch dimensions reinterpreted as event
#'   dimensions (model building only). Defaults to `0`.
#' @param create_obj Logical. If `TRUE`, return the distribution object itself
#'   instead of a random variable (e.g. to build mixtures). Defaults to `FALSE`.
#' @param to_jax Logical. Return a JAX array (`TRUE`, default) or convert to R.
#'
#' @return
#' \itemize{
#'   \item When `sample = FALSE`: A BF GaussianStateSpace distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     GaussianStateSpace distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.gaussian_state_space(
#'     num_steps = 5L,
#'     transition_matrix = jnp$array(matrix(c(0.5, 0.1, 0.0, 0.8), nrow = 2L, byrow = TRUE)),
#'     covariance_matrix = jnp$array(matrix(c(1.0, 0.6, 0.6, 1.0), nrow = 2L, byrow = TRUE)),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#gaussianstate}
#'
#' @export
bf.dist.gaussian_state_space=function(num_steps, transition_matrix, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     num_steps=as.integer(num_steps);
     if(!.BF_env$.py$is_none(transition_matrix)){transition_matrix = .BF_env$jnp$asarray(transition_matrix)}
     if(!.BF_env$.py$is_none(covariance_matrix)){covariance_matrix = .BF_env$jnp$asarray(covariance_matrix)}
     if(!.BF_env$.py$is_none(precision_matrix)){precision_matrix = .BF_env$jnp$asarray(precision_matrix)}
     if(!.BF_env$.py$is_none(scale_tril)){scale_tril = .BF_env$jnp$asarray(scale_tril)}


      .BF_env$.bf_instance$dist$gaussian_state_space(
        num_steps,
        transition_matrix = transition_matrix,
        covariance_matrix= covariance_matrix,
        precision_matrix= precision_matrix,
        scale_tril= scale_tril,
        validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title Multivariate Student's t distribution
#'
#' @description
#' The Multivariate Student's t distribution is a generalization of the
#' Student's t distribution to multiple dimensions. It is a heavy-tailed
#' distribution that is often used to model data that is not normally
#' distributed.
#'
#' The PDF of the multivariate Student's t-distribution for a random vector (
#' \\mathbf\{x\} \\in \\mathbb\{R\}^d ) is given by:
#'
#' \deqn{f(\mathbf{x}) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \nu^{d/2} \pi^{d/2} | \Sigma|^{1/2}} \left(1 + \frac{1}{\nu} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)^{-(\nu + d)/2}}
#'
#' where:
#'
#' * \eqn{\Gamma(\cdot)} is the Gamma function.
#' * \eqn{\mu} is the mean vector.
#' * \eqn{\Sigma} is the scale (covariance) matrix.
#' * \eqn{\nu} is the degrees of freedom.
#' * \eqn{d} is the dimensionality of \eqn{\mathbf{x}}.
#'
#' @param df Degrees of freedom, must be positive.
#' @param loc Location vector, representing the mean of the distribution.
#'   scale_tril (numeric vector or array): Lower triangular matrix defining the
#'   scale. - `shape` (integer vector): A multi-purpose argument for shaping. When
#'   `sample = FALSE` (model building), this is used with `shape` to set the
#'   distribution's batch shape. When `sample = TRUE` (direct sampling), this is
#'   used as `sample_shape` to draw a raw JAX array of the given shape.
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
#'   \item When `sample = FALSE`: A BF MultivariateStudentT distribution object
#'     (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     MultivariateStudentT distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.multivariate_student_t(df = 2, loc = jnp$array(c(1.0, 0.0, -2.0)), scale_tril = jnp$linalg$cholesky(jnp$array(matrix(c(2.0, 0.7, -0.3, 0.7, 1.0, 0.5, -0.3, 0.5, 1.5), nrow = 3L, byrow = TRUE))), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multivariatestudentt}
#'
#' @references
#' [Multivariate Student-t distribution –
#' Wikipedia](https://en.wikipedia.org/wiki/Multivariate_Student%27s_t-distribution)
#'
#' @export
bf.dist.multivariate_student_t=function(df, loc=0.0, scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$multivariate_student_t(
       df = .BF_env$jnp$asarray(df),
       loc = .BF_env$jnp$asarray(loc),
       scale_tril = .BF_env$jnp$asarray(scale_tril),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

#' @title  Levy distribution.
#'
#' @description
#' The Lévy distribution is a continuous probability distribution on the positive real line (or shifted positive line)
#' that is heavy-tailed, skewed, and arises naturally in connection with stable distributions
#' specifically the case with stability index \deqn{\alpha = \tfrac12}.
#' It is often used in contexts such as hitting-time problems for Brownian motion, physics (e.g., van der Waals line-shapes),
#' and modelling very heavy-tailed phenomena. Let (X) be a Lévy-distributed random variable with location parameter \deqn{\mu}
#' and scale parameter (c > 0). The support is \deqn{x \ge \mu}.
#' @param loc A numeric vector, matrix, or array representing the location parameter.
#' @param scale A numeric vector, matrix, or array representing the scale parameter.
#' @param shape A numeric vector used for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
#' @param to_jax Logical. Defaults to TRUE.
#'
#'  @return
#'  - When \code{sample=FALSE}, a BI Levy distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Levy distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#levy}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m <- importBI(platform = "cpu")
#' bi.dist.levy(loc = 1, scale = 10, sample = TRUE)
#' }
#' @export

bi.dist.levy <- function(loc, scale, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BI_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BI_env$.bi_instance$dist$levy(
    loc = .BI_env$jnp$array(loc),
    scale = .BI_env$jnp$array(scale),
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

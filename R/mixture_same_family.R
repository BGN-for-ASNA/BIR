#' @title Finite mixture of component distributions from the same family. distribution
#'
#' @description
#' A *Mixture (Same-Family)* distribution is a finite mixture in which **all
#' components come from the same parametric family** (for example, all Normal
#' distributions but with different parameters), and are combined via mixing
#' weights. The class is typically denoted as:
#'
#' Let the mixture have (K) components. Let weights (w_i\\ge0), (\\sum_\{i=1\}^K
#' w_i = 1). Let the component family have density (f(x \\mid \\theta_i)) for
#' each component (i). Then the mixture’s PDF is
#'
#' \deqn{f_X(x) = \sum_{i=1}^K w_i ; f(x \mid \theta_i).}
#'
#' where each \eqn{f(x \mid \theta_i)} is from the same family with parameter
#' \eqn{\theta_i}.
#'
#' @param mixing_distribution A `Categorical` distribution representing the
#'   mixing weights.
#' @param component_distribution See the Python documentation of
#'   `component_distribution`.
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
#'   \item **When `sample = FALSE`**: A BF MixtureSameFamily distribution object
#'     (for model building).
#'   \item **When `sample = TRUE`**: A JAX array of samples drawn from the
#'     MixtureSameFamily distribution (for direct sampling).
#'   \item **When `create_obj = TRUE`**: The raw BF distribution object (for
#'     advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.mixture_same_family(mixing_distribution = bf.dist.categorical(probs = jnp$array(c(0.3, 0.7)), create_obj = TRUE), component_distribution = bf.dist.normal(loc = 0.0, scale = 1.0, shape = c(2L), create_obj = TRUE), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#mixture-,}
#'
#' @references
#' [Mixture distributions
#' ](https://cseweb.ucsd.edu/~elkan/250Bwinter2011/mixturemodels.pdf)
#'
#' @export
bf.dist.mixture_same_family <- function(mixing_distribution, component_distribution, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(), sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

  shape <- do.call(tuple, as.list(as.integer(shape)))
  reticulate::py_run_string("def is_none(x): return x is None")
  if (!.BF_env$.py$is_none(seed)) {
    seed <- as.integer(seed)
  }
  .BF_env$.bf_instance$dist$mixture_same_family(
    mixing_distribution,
    component_distribution,
    validate_args = validate_args, name = name, obs = obs, mask = mask, sample = sample, seed = seed, shape = shape, event = event, create_obj = create_obj, to_jax = to_jax
  )
}

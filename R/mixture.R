#' @title A marginalized finite mixture of component distributions. distribution
#'
#' @description
#' A **marginalised finite mixture of component distributions** refers to a
#' probability model in which you have a finite number \eqn{K} of component
#' distributions (e.g., normals, Poissons, etc.), each weighted by a mixing
#' probability from a Categorical distribution. We **marginalise out** the
#' latent assignment variable (i.e., you don’t explicitly model which component
#' each data point came from). The resulting distribution can be either a
#' `mixture_general` (when component distributions are a list) or a
#' `mixture_same_Family` (when component distributions are a single
#' distribution).
#'
#' Suppose you have:
#'
#' * Mixing weights \eqn{\pi_1, \dots, \pi_K}, with \eqn{\pi_j \ge 0} and
#' \eqn{\sum_{j=1}^K \pi_j = 1}.
#' * Component distributions (densities or PMFs) \eqn{f_j(x \mid \theta_j)}, for
#' \eqn{j = 1,\dots,K}.
#'
#' Then the marginalised mixture distribution for an observed \eqn{X} is:
#'
#' \deqn{p(X = x) = \sum_{j=1}^K \pi_j ; f_j(x \mid \theta_j). $ This is exactly the mixture density/PMF you get when you marginalise the latent component‑assignment variable $Z \in {1,\dots,K}$ out of the joint distribution:}
#'
#' p(x) = \\sum_\{z=1\}^K p(z = j); p(x \\mid z = j) = \\sum_\{j=1\}^K \\pi_j ;
#' f_j(x \\mid \\theta_j). $$
#'
#' @param mixing_distribution The distribution used to determine the mixing
#'   weights.
#' @param component_distributions The list of component distributions.
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
#'   \item When `sample = FALSE`: A BF Mixture distribution object (for model
#'   \item building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Mixture
#'   \item distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'   \item use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.mixture(mixing_distribution = bf.dist.categorical(probs = jnp$array(c(0.3, 0.7)), create_obj = TRUE), component_distributions = list(bf.dist.normal(0, 1, create_obj = TRUE), bf.dist.normal(2, 1, create_obj = TRUE)), sample = TRUE, shape = c(10L))
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#mixture}
#'
#' @references
#' https://en.wikipedia.org/wiki/Mixture_distribution "Mixture distribution"
#'
#' https://pymc3-testing.readthedocs.io/en/rtd-docs/notebooks/marginalized_gaussian_mixture_model.html
#' "Marginalized Gaussian Mixture Model — PyMC3 3.1rc3 documentation"
#'
#' @export
bf.dist.mixture=function(mixing_distribution, component_distributions, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$mixture(
       mixing_distribution,
       component_distributions = reticulate::r_to_py(component_distributions, convert = TRUE),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

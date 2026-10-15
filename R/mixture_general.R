#' @title Mixture General distribution
#'
#' @description
#' A mixture distribution is a probability distribution constructed by selecting
#' one of several component distributions according to specified weights, and
#' then drawing a sample from the chosen component. It allows modelling of
#' heterogeneous populations and multimodal data.
#'
#' Let the mixture have (K) component distributions with densities (f_i(x)) for
#' (i = 1,\\dots,K). Let weights       (w_i\\ge0) satisfy (\\sum_\{i=1\}^K w_i =
#' 1). Then the PDF is
#'
#' \deqn{f(x) = \sum_{i=1}^K w_i ; f_i(x).}
#'
#' **Parameters:**
#'
#' *   **mixing_distribution**:  A `Categorical` distribution representing the
#' mixing weights.
#'
#' *   **component_distributions**: A list of distributions representing the
#' components of the mixture.
#'
#' *   **sample (bool, optional): A control-flow argument. If `TRUE`, the
#' function will directly sample a raw JAX array from the distribution,
#' bypassing the BF model context. If `FALSE`, it will create a `BF.sample` site
#' within a model. Defaults to `FALSE`.
#'
#' * `shape` (integer vector): A multi-purpose argument for shaping. When
#' `sample = FALSE` (model building), this is used with `shape` to set the
#' distribution's batch shape. When `sample = TRUE` (direct sampling), this is
#' used as `sample_shape` to draw a raw JAX array of the given shape.
#'
#' *   **seed (int, optional)**:  An integer seed for this draw when `sample =
#' TRUE`. By default (`py_none()`) the draw comes from the session's random
#' stream, set by `bf(rand_seed=...)`, so consecutive draws differ but the whole
#' sequence is reproducible. No effect when `sample = FALSE`.
#'
#' *   **obs (numeric vector or array, optional)**:  The observed value for this
#' random variable. If provided, the sample site is conditioned on this value,
#' and the function returns the observed value. If `py_none()`, the site is
#' treated as a latent (unobserved) random variable. Defaults to `py_none()`.
#'
#' *   **name (str, optional)**:  The name of the sample site in a BF model.
#' This is used to uniquely identify the random variable. Defaults to 'x'.
#'
#' **#### Returns:**
#'
#' *   **When `sample = FALSE`**: A BF MixtureGeneral distribution object (for
#' model building).
#' *   **When `sample = TRUE`**: A JAX array of samples drawn from the
#' MixtureGeneral distribution (for direct sampling).
#' *   **When `create_obj = TRUE`**: The raw BF distribution object (for
#' advanced use cases).
#'
#' **#### Example Usage:**
#'
#' from BayesForge import bf m = bf(platform='cpu') bf.dist.mixture_general(
#' mixing_distribution=bf.dist.categorical(probs=jnp.array(\[0.3, 0.7\]),
#' create_obj = TRUE), component_distributions=\[bf.dist.normal(loc=0.0,
#' scale=1.0, create_obj = TRUE),bf.dist.normal  (loc=0.0, scale=1.0,
#' create_obj = TRUE)\], sample = TRUE )
#'
#' @param mixing_distribution See the Python documentation of
#'   `mixing_distribution`.
#' @param component_distributions See the Python documentation of
#'   `component_distributions`.
#' @param support See the Python documentation of `support`.
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
#' @return A random variable (model building) or draws (`sample = TRUE`).
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.mixture_general(mixing_distribution = bf.dist.categorical(probs = jnp$array(c(0.3, 0.7)), create_obj = TRUE), component_distributions = list(bf.dist.normal(-2.0, 1.0, create_obj = TRUE), bf.dist.normal(2.0, 1.0, create_obj = TRUE)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#mixturegeneral}
#'
#' @references
#' \[1\]:
#' https://preliz.readthedocs.io/en/latest/distributions/gallery/mixture.html
#' "Mixture Distribution — PreliZ 0.22.0 documentation"
#'
#' \[2\]: https://reliability.readthedocs.io/en/latest/Mixture\%20models.html
#' "Mixture models — reliability 0.9.0 documentation"
#'
#' \[3\]:
#' https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Mixture.html
#' "Mixture — SciPy v1.16.2 Manual"
#'
#' @export
bf.dist.mixture_general=function(mixing_distribution, component_distributions, support=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$mixture_general(
       mixing_distribution,
       component_distributions = reticulate::r_to_py(component_distributions, convert = TRUE),
       support= support,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

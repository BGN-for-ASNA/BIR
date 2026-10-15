#' @title Dirichlet-Multinomial distribution
#'
#' @description
#' Creates a Dirichlet-Multinomial compound distribution, which is a Multinomial
#' distribution with a Dirichlet prior on its probabilities. It is often used in
#' Bayesian statistics to model count data where the proportions of categories
#' are uncertain.
#'
#' The **Dirichlet-Multinomial** distribution describes a model in which you
#' first draw a probability vector
#'
#' \deqn{\mathbf{p} = (p_1, \dots, p_K) \sim \mathrm{Dirichlet}(\alpha)}
#'
#' and then you draw counts
#'
#' \deqn{\mathbf{x} = (x_1, \dots, x_K) \sim \mathrm{Multinomial}(n, \mathbf{p}).}
#'
#' Integrating out \eqn{\mathbf{p}} gives you the marginal distribution of
#' \eqn{\mathbf{x}}.
#' * It generalizes the **Beta-Binomial** model to multiple categories (i.e.
#' \eqn{K > 2}).
#' * A key feature is that it introduces **overdispersion** relative to a
#' standard multinomial: counts are more “clumped” because the underlying
#' probabilities vary according to the Dirichlet prior.
#'
#' @param concentration The concentration parameter (alpha) for the Dirichlet
#'   prior. Values must be positive. The last dimension is interpreted as the
#'   number of categories.
#' @param total_count The total number of trials (n). This must be a
#'   non-negative integer. Defaults to 1.
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
#' @param infer Optional named list passed to NumPyro's `infer` argument (e.g.
#'   `list(enumerate = "parallel")`).
#' @param enum Logical. If `TRUE`, the discrete latent is summed out exactly
#'   (parallel enumeration) instead of sampled; same as `infer = list(enumerate =
#'   "parallel")`. Defaults to `FALSE`.
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
#'   \item When `sample = FALSE`: A BF dirichlet_multinomial Distribution
#'     distribution object (for  model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     dirichlet_multinomial distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF dirichlet_multinomial Distribution
#'     object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' counts <- bf.dist.dirichlet_multinomial(concentration = jnp$array(c(1.0, 1.0, 1.0)), total_count = 10L, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#dirichletmultinomial}
#'
#' @references
#' https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
#'
#' https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html
#'
#' @export
bf.dist.dirichlet_multinomial=function(concentration, total_count=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)

     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$dirichlet_multinomial(
       concentration = .BF_env$jnp$asarray(concentration),
       total_count = .BF_env$jnp$asarray(as.integer(total_count)),
       validate_args = validate_args,
       name= name,
       obs= obs,
       mask= mask,  infer= .bf_infer(infer, enum),
       sample= sample,
       seed= seed,
       shape= shape,
       event= event,
       create_obj= create_obj,
       to_jax = to_jax
       )
}

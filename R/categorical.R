#' @title Categorical distribution. distribution
#'
#' @description
#' The Categorical distribution, also known as the multinomial distribution,
#' describes the probability of \eqn{K} different outcomes with probabilities
#' \eqn{{p_1, p_2, \dots, p_K}}.There’s no inherent ordering required among the
#' categories.
#'
#' - The probabilities satisfy (p_i \\ge 0) for all (i) and \eqn{\sum_{i=1}^K
#' p_i = 1} .
#' - It’s essentially the distribution of a single draw from a discrete set of
#' categories.
#' - It generalises the Bernoulli distribution (which is the special case
#' \eqn{K=2}).
#'
#' It is commonly used to model discrete choices or classifications.
#'
#' If \eqn{X} is a categorical random variable taking values in \eqn{{1, 2,
#' \dots, K}} with probabilities \eqn{p_1, \dots, p_K}, then:
#'
#' \deqn{\Pr(X = i) = p_i, \quad i = 1,\dots, K.}
#'
#' where \eqn{p_k} is the probability of outcome \eqn{k}, and the sum is over
#' all possible outcomes.
#'
#' @param probs A 1D array of probabilities for each category. Must sum to 1.
#' @param logits A 1D array of unnormalized log probabilities for each category.
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
#'   \item When `sample = FALSE`: A BF Categorical distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Categorical
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.categorical(probs = jnp$array(c(0.2, 0.3, 0.5)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#categorical}
#'
#' @references
#' https://en.wikipedia.org/wiki/Categorical_distribution"Categorical
#' distribution"
#'
#' https://distribution-explorer.github.io/discrete/categorical.html"Categorical
#' distribution — Probability Distribution Explorer documentation"
#'
#' https://static.hlt.bme.hu/semantics/external/pages/n-gram/en.wikipedia.org/wiki/Categorical_distribution.html"Categorical
#' distribution - Wikipedia"
#'
#' @export
bf.dist.categorical=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     reticulate::py_run_string("def is_none(x): return x is None")
     if(!.BF_env$.py$is_none(logits)){logits = .BF_env$jnp$asarray(logits)}
     if(!.BF_env$.py$is_none(probs)){probs = .BF_env$jnp$asarray(probs)}
     .BF_env$.bf_instance$dist$categorical(
       probs = probs,
       logits = logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}




#' @title Negative Binomial distribution
#'
#' @description
#' The Negative Binomial distribution models the number of failures (or the
#' total number of trials) in a sequence of independent Bernoulli trials with
#' success probability (p), until a specified number `total_count` (r) of
#' successes is achieved. It is often used as a count-data model when the
#' variance exceeds the mean (“overdispersion”) relative to a Poisson.
#'
#' PMF (Probability Mass Function): Given (X) = number of failures before the
#' (r)th success (so (X = 0,1,2,\\dots)), with (r) successes fixed, success
#' probability (p), failure probability (q = 1 - p):
#'
#' \deqn{\Pr(X = k) ;=; \binom{k + r - 1}{k} ; p^r ; q^k, \quad k = 0,1,2,\dots}
#'
#' @param total_count The total number of events.
#' @param probs See the Python documentation of `probs`.
#' @param logits See the Python documentation of `logits`.
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
#'   \item BF NegativeBinomial distribution object (for model building).
#'   \item JAX array of samples drawn from the NegativeBinomial distribution (for
#'     direct sampling).
#'   \item The raw BF distribution object (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.negative_binomial(total_count = 5.0, probs = jnp$array(c(0.2, 0.3, 0.5)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#negativebinomial}
#'
#' @references
#' https://en.wikipedia.org/wiki/Negative_binomial_distribution
#'
#' https://www.statology.org/negative-binomial-distribution/
#'
#' @export
bf.dist.negative_binomial=function(total_count, probs, logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if (.BF_env$.py$is_none(logits)){
       .BF_env$.bf_instance$dist$negative_binomial(total_count = .BF_env$jnp$asarray(total_count), probs=.BF_env$jnp$asarray(probs),   validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
     }else{
       .BF_env$.bf_instance$dist$negative_binomial(total_count = .BF_env$jnp$asarray(total_count), logits= .BF_env$jnp$asarray(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)

     }
}

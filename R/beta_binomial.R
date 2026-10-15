#' @title BetaBinomial distribution
#'
#' @description
#' Samples from a BetaBinomial distribution, a compound distribution where the
#' probability of success in a binomial experiment is drawn from a Beta
#' distribution. This models situations where the underlying probability of
#' success is not fixed but varies according to a prior belief represented by
#' the Beta distribution. It is often used to model over-dispersion relative to
#' the binomial distribution.
#'
#' For \eqn{X \sim \mathrm{BetaBinomial}(N,\alpha,\beta)}, for \eqn{k =
#' 0,1,\dots,N}:
#'
#' \deqn{\Pr(X = k) = \binom{N}{k} ; \frac{B(k + \alpha,; N - k + \beta)}{B(\alpha,\beta)},}
#'
#' where (\eqn{B(u,v)}) is the Beta function \eqn{(B(u,v) =
#' \frac{\Gamma(u),\Gamma(v)}{\Gamma(u+v)})}.
#'
#' @param concentration1 The first concentration parameter (alpha) of the Beta
#'   distribution.
#' @param concentration0 The second concentration parameter (beta) of the Beta
#'   distribution.
#' @param total_count The number of Bernoulli trials in the Binomial part of the
#'   distribution.
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
#'   \item When `sample = FALSE`: A BF BetaBinomial distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     BetaBinomial distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF BetaBinomial distribution object
#'     (for advanced use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.beta_binomial(concentration1 = 1.0, concentration0 = 1.0, total_count = 10L, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#betabinomial}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Beta-binomial_distribution
#' "Beta-binomial distribution"
#'
#' \[2\]: https://www.statisticshowto.com/beta-binomial-distribution/
#' "Beta-Binomial Distribution: Definition - Statistics How To"
#'
#' \[3\]:
#' https://mc-stan.org/docs/2_19/functions-reference/beta-binomial-distribution.html
#' "12.3 Beta-Binomial Distribution | Stan Functions Reference"
#'
#' @export
bf.dist.beta_binomial=function(concentration1, concentration0, total_count=1, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$beta_binomial(
       concentration1 = .BF_env$jnp$asarray(concentration1),
       concentration0 = .BF_env$jnp$asarray(concentration0),
       total_count = .BF_env$jnp$asarray(as.integer(total_count)),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}



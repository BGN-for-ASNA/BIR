#' @title Binomial distribution
#'
#' @description
#' The Binomial distribution models the number of successes in a sequence of
#' independent Bernoulli trials. It represents the probability of obtaining
#' exactly *k* successes in *n* trials, where each trial has a probability *p*
#' of success. It assumes:
#'
#' 1. Each trial has exactly two possible outcomes (success or failure).
#' 2. The probability of success $p is constant across trials.
#' 3. Trials are independent.
#'
#' We denote:
#'
#' \deqn{X \sim \mathrm{Binomial}(n, p).}
#'
#' ## Probability Mass Function (PMF)
#'
#' For \eqn{k = 0,1,2,\dots,n}:
#'
#' \deqn{\Pr(X = k) = \binom{n}{k} ; p^k ; (1 - p)^{,n - k},}
#'
#' where \eqn{\binom{n}{k} = \frac{n!}{k!(n-k)!}}.
#'
#' This formula counts the number of ways to choose which \eqn{k} out of \eqn{n}
#' trials are the successes (the binomial coefficient) and then multiplies by
#' the probability of each such configuration \eqn{p^k (1-p)^{n-k}}.
#'
#' @param total_count The number of trials *n*.
#' @param probs The probability of success *p* for each trial. Must be between 0
#'   and 1.
#' @param logits The log-odds of success for each trial.  `probs =
#'   jax.nn.sigmoid(logits)`.
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
#'   \item Binomial distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the Binomial distribution (for direct
#'     sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.binomial(total_count = 10L, probs = 0.5, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#binomial}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Binomial_distribution "Binomial
#' distribution"
#'
#' \[2\]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm
#' "1.3.6.6.18. Binomial Distribution - Information Technology Laboratory"
#'
#' @export
bf.dist.binomial=function(
    total_count=1L,
    probs=py_none(),
    logits=py_none(),
    validate_args=py_none(),
    name='x',
    obs=py_none(),
    mask=py_none(), infer=py_none(), enum=FALSE,
    sample=FALSE,
    seed=py_none(),
    shape=c(),
    event=0,
    create_obj=FALSE,
    to_jax = TRUE
    ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){
       if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     }

     if (.BF_env$.py$is_none(logits)){
       .BF_env$.bf_instance$dist$binomial(
         total_count=.BF_env$jnp$asarray(total_count),
         probs= .BF_env$jnp$asarray(probs),
         validate_args= validate_args,
         name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),
         sample= sample,  seed= seed,  shape= shape,
         event= event,  create_obj= create_obj,   to_jax = to_jax)
     }else{
       .BF_env$.bf_instance$dist$binomial(total_count=.BF_env$jnp$asarray(total_count),  logits= .BF_env$jnp$asarray(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj, to_jax = to_jax)

     }
}



#' @rdname bf.dist.binomial
#' @description `bf.dist.bfnomial()` is a deprecated alias of `bf.dist.binomial()`.
#' @param ... Arguments passed to `bf.dist.binomial()`.
#' @export
bf.dist.bfnomial <- function(...) {
  .Deprecated("bf.dist.binomial")
  bf.dist.binomial(...)
}

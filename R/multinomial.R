#' @title Multinomial distribution
#'
#' @description
#' The multinomial distribution is a discrete probability distribution that
#' generalizes the Binomial distribution to the case of **more than two possible
#' outcomes per trial**. Specifically:
#'
#' * You perform a fixed number \eqn{n} of independent trials.
#' * On each trial there are \eqn{k} possible outcomes, labeled (1,2,\\dots,k).
#' * The probability of outcome \eqn{i} on any trial is (p_i), where (p_1 + p_2 + \\cdots + p_k = 1).
#' * Let \eqn{X_i} be the number of trials (out of \eqn{n}) that produced
#' outcome \eqn{i}. Then the vector \eqn{\mathbf X = (X_1, \dots, X_k)} follows
#' a multinomial distribution.
#'
#' It reduces to the binomial when (k=2).
#'
#' PMF (Probability Mass Function):
#'
#' For \eqn{\mathbf x = (x_1,\dots,x_k)} with each \eqn{x_i \ge 0} integer and
#' \eqn{\sum_{i=1}^k x_i = n}, $$ \\Pr(X_1 = x_1, \\dots, X_k = x_k) =
#' \\frac\{n!\}\{x_1!,x_2!\\cdots x_k!\} ; p_1^\{,x_1\}; p_2^\{,x_2\} \\cdots
#' p_k^\{,x_k\}. $ If \eqn{\sum_i x_i \neq n}, the probability is 0.
#'
#' @param total_count The number of trials.
#' @param probs Event probabilities. Must sum to 1.
#' @param logits Event log probabilities. - *total_count_max (int, optional): An
#'   optional integer providing an upper bound on `total_count`. This is used for
#'   performance optimization with `lax.scan` when `total_count` is a dynamic JAX
#'   tracer, helping to avoid recompilation.
#' @param total_count_max See the Python documentation of `total_count_max`.
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
#'   \item When `sample = FALSE`: A BF Multinomial distribution object (for model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the Multinomial
#'     distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.multinomial(total_count = 10L, probs = jnp$array(c(0.2, 0.3, 0.5)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multinomial}
#'
#' @references
#' https://online.stat.psu.edu/stat504/book/export/html/667 "2.3 - The
#' Multinomial Distribution - STAT ONLINE"
#'
#' https://en.wikipedia.org/wiki/Multinomial_distribution "Multinomial
#' distribution"
#'
#' https://faculty.washington.edu/yenchic/20A_stat512/Lec7_Multinomial.pdf
#' "\[PDF\] Lecture 7: Multinomial distribution"
#'
#' @export
bf.dist.multinomial=function(total_count=1, probs=py_none(), logits=py_none(), total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(total_count_max)){total_count_max=as.integer(total_count_max);}
     if(!.BF_env$.py$is_none(logits)){logits= .BF_env$jnp$asarray(logits)}
     if(!.BF_env$.py$is_none(probs)){probs= .BF_env$jnp$asarray(probs)}
     .BF_env$.bf_instance$dist$multinomial(total_count=total_count,  probs= probs, logits= logits, total_count_max= total_count_max,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

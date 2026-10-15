#' @title Ordered Logistic distribution
#'
#' @description
#' The ordered logistic distribution is used for modeling **ordinal** outcome
#' variables \eqn{Y \in {1,2,\dots,K}} (i.e., categories with a natural order)
#' via a latent continuous predictor \eqn{\eta} and a set of increasing
#' *cut-points* \eqn{c_1 < c_2 < \cdots < c_{K-1}}. When \eqn{\eta} crosses
#' thresholds, the observed  \eqn{Y}
#'
#' ### PMF (Probability Mass Function)
#'
#' Let \eqn{\eta \in \mathbb R} be the linear predictor (for example \eqn{\eta =
#' x^\top \beta}), let cutpoints \eqn{c = (c_1, \dots,c_{K-1})} satisfy \eqn{c_1
#' < c_2 < \cdots < c_{K-1}}. Define also \eqn{c_0 = -\infty}, \eqn{c_K =
#' +\infty}. Then for (k \\in \{1,\\dots,K\}),
#'
#' \deqn{\Pr(Y = k \mid \eta, c) = \begin{cases} 1 - \mathrm{logit}^{-1}(\eta - c_1), & k = 1, [6pt] \mathrm{logit}^{-1}(\eta - c_{k-1}) - \mathrm{logit}^{-1}(\eta - c_{k}), & 1 < k < K, [6pt] \mathrm{logit}^{-1}(\eta - c_{K-1}), & k = K. \end{cases}}
#'
#' Here \eqn{\mathrm{logit}^{-1}(z) = \frac{1}{1 + e^{-z}}}. (\[Stan\]\[2\]) In
#' other words:
#'
#' \deqn{\Pr(Y \le k \mid \eta, c) = \mathrm{logit}^{-1}(\eta - c_k),}
#'
#' and so
#'
#' \deqn{\Pr(Y = k) = \Pr(Y \le k) - \Pr(Y \le k-1).}
#'
#' This formulation appears in e.g. the CDF‐link setup of the proportional odds
#' model.
#'
#' @param predictor Prediction in real domain; typically this is output of a
#'   linear model.
#' @param cutpoints Positions in real domain to separate categories.
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
#'   \item When `sample = FALSE`: A BF OrderedLogistic distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     OrderedLogistic distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.ordered_logistic(predictor = jnp$array(c(0.2, 0.5, 0.8)), cutpoints = jnp$array(c(-1.0, 0.0, 1.0)), sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#orderedlogistic}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Ordered_logit
#'
#' \[2\]:
#' https://mc-stan.org/docs/2_21/functions-reference/ordered-logistic-distribution.html
#'
#' \[3\]: https://labdisia.disia.unifi.it/grilli/files/Papers/Ordered_Logit.pdf"
#'
#' @export
bf.dist.ordered_logistic=function(predictor, cutpoints, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$ordered_logistic(
       predictor = .BF_env$jnp$asarray(predictor),
       cutpoints = .BF_env$jnp$asarray(cutpoints),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

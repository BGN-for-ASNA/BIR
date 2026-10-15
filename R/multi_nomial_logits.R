#' @title Multinomial Logits distribution
#'
#' @description
#' A *multinomial logits* distribution refers to a categorical (or more
#' generally multinomial) distribution over (K) classes whose probabilities are
#' given via the softmax of a vector of logits. That is, given a vector of
#' real-valued logits \eqn{\ell = (\ell_1, …, \ell_K)}, the class probabilities
#' are:
#'
#' \deqn{p_k = \frac{\exp(\ell_k)}{\sum_{j=1}^K \exp(\ell_j)}.}
#'
#' Then a single draw from the distribution yields one of the (K) classes (or
#' for a multinomial count version, counts over the classes) with those
#' probabilities.
#'
#' **Categorical version (one draw):** Let (X \\in \{1,2,\\dots,K\}) be the
#' class. Then
#'
#' \deqn{\Pr(X=k \mid \ell) ;=; p_k = \frac{\exp(\ell_k)}{\sum_{j=1}^K \exp(\ell_j)}.}
#'
#' This directly uses the logits \eqn{\ell}.
#'
#' **Multinomial version (n draws):** If you draw (n) independent draws (or
#' equivalently count vector) with probabilities (p = (p_1,…,p_K)), then for a
#' count vector (x = (x_1,…,x_K)) with (\\sum_k x_k = n),
#'
#' \deqn{\Pr(X = x \mid n, p) = \frac{n!}{x_1! \cdots x_K!} ; \prod_{k=1}^K p_k^{x_k}}
#'
#' and \eqn{p_k} is given by the softmax of logits.
#'
#' @param logits Logits for each outcome. Must be at least one-dimensional.
#' @param total_count The total number of trials.
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
#'   \item When `sample = FALSE`: A BF MultinomialLogits distribution object (for
#'     model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     MultinomialLogits distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.multinomial_logits(logits = jnp$array(c(1.0, 0.5)), total_count = 5L, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#multinomiallogits}
#'
#' @references
#' https://en.wikipedia.org/wiki/Multinomial_distribution
#'
#' @export
bf.dist.multinomial_logits=function(logits, total_count=1, total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!.BF_env$.py$is_none(total_count_max)){total_count_max= as.integer(total_count_max)}
     total_count= .BF_env$jnp$asarray(as.integer(total_count))

     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$multinomial_logits(
       logits = .BF_env$jnp$asarray(logits),
       total_count = .BF_env$jnp$asarray(total_count),
       total_count_max = total_count_max,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

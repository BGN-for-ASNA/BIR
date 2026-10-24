#' @title Generic Zero Inflated distribution
#'
#' @description
#' A Zero-Inflated distribution combines a base distribution with a Bernoulli
#' distribution to model data with an excess of zero values. It assumes that
#' each observation is either drawn from the base distribution or is a zero with
#' probability determined by the Bernoulli distribution (the "gate"). A
#' **zero-inflated distribution** arises when you take a random variable \eqn{X}
#' that originally has some distribution (with PMF \eqn{f_X(x)} ) *and* you add
#' extra mass at zero. In other words you only observe \eqn{X} when \eqn{X \ge
#' 0} (if count) and you assume there are more zeros than what the original
#' distribution predicts — so you mix in a point-mass at zero. A zero-inflated
#' model assumes two processes: a “structural zero” process with probability
#' \eqn{\pi}, and another process (the base count/distribution) with probability
#' \eqn{1 - \pi}, which itself may generate zeros *or* non-zeros.
#'
#' - When / Why use it:
#' * **Data collection constraints**: You can only observe values in a process
#' when some condition is met; others are automatically zero (for example, when
#' the event cannot happen for some units).
#' * **Mixtures of processes**: Some observations are “structural zeros” (no
#' risk) and others follow a regular count process (with risk).
#' * **Overdispersion & excess zeros**: If you try a standard count distribution
#' (Poisson, Negative Binomial) and you observe many more zeros than predicted
#' (given the non-zero counts), a zero-inflated alternative may fit better.
#'
#' * The general PMF form is (for discrete count models):
#'
#' \deqn{P(X=0) = \pi + (1 - \pi),P_{\text{base}}(0), \quad P(X=k) = (1 - \pi),P_{\text{base}}(k), ; k > 0.}
#'
#' @param base_dist The base distribution to be zero-inflated (e.g., Poisson,
#'   NegativeBinomial).
#' @param gate Probability of extra zeros (between 0 and 1).
#' @param gate_logits Log-odds of extra zeros.
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
#'   \item When `sample = FALSE`: A BF ZeroInflatedDistribution distribution
#'     object (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     ZeroInflatedDistribution distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.zero_inflated_distribution(
#'     base_dist = bf.dist.poisson(rate = 5, create_obj = TRUE),
#'     gate = 0.3,
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Zero-inflated_model "Zero-inflated
#' model"
#'
#' \[2\]:
#' https://larmarange.github.io/guide-R/analyses_avancees/modeles-zero-inflated.html
#' "48 Modèles de comptage zero-inflated et hurdle – guide-R"
#'
#' @export
bf.dist.zero_inflated_distribution=function(base_dist, gate=py_none(), gate_logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if(!.BF_env$.py$is_none(gate)){gate = .BF_env$jnp$asarray(gate)}
     if(!.BF_env$.py$is_none(gate_logits)){gate_logits = .BF_env$jnp$asarray(gate_logits)}

     .BF_env$.bf_instance$dist$zero_inflated_distribution(
       base_dist,
       gate =  .BF_env$jnp$asarray(gate),
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

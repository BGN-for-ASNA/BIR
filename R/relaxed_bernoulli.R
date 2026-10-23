#' @title Relaxed Bernoulli distribution
#'
#' @description
#' The Relaxed Bernoulli is a continuous distribution on the interval
#' \eqn{(0,1)} that smoothly approximates the discrete Bernoulli distribution
#' (which has support \eqn{{0,1}}). It was introduced to allow for
#' *differentiable* sampling of approximate binary random variables, which is
#' useful in variational inference and other gradient‐based optimization
#' settings.
#'
#' The probability density function (PDF) is defined as:
#'
#' * It has a *temperature* parameter \eqn{\tau > 0} controlling the smoothness:
#' as \eqn{\tau \to 0}, the distribution concentrates near \eqn{{0, 1}} (thus
#' approximating the discrete Bernoulli).
#' * For larger \eqn{\tau}, the sample becomes more “soft” and less binary,
#' e.g., converging toward 0.5 for very high \eqn{\tau}.
#' * It is parameterised by a probability \eqn{p} (or equivalently logits
#' \eqn{\ell = \log\frac{p}{1-p}}).
#'
#' Let \eqn{X \in (0,1)} follow a Relaxed Bernoulli (Binary Concrete)
#' distribution with location parameter \eqn{\alpha > 0} (sometimes expressed
#' via logits) and temperature parameter \eqn{\lambda > 0}. Then the PDF is:
#'
#' \deqn{p_{,\alpha,;\lambda}(x) = \frac{;\lambda,\alpha,x^{-\lambda - 1},(1 - x)^{-\lambda - 1};} {\Bigl(\alpha,x^{-\lambda} + (1 - x)^{-\lambda}\Bigr)^{2}}, \quad \text{for } 0 < x < 1.}
#'
#' @param temperature The temperature parameter. Must be greater than 0.
#' @param probs The probability of success. Must be in the interval `\[0, 1\]`.
#'   Only one of `probs` or `logits` can be specified.
#' @param logits The log-odds of success. Must be in the interval `\[-inf,
#'   inf\]`. Only one of `probs` or `logits` can be specified.
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
#'   \item BF RelaxedBernoulli distribution object (for model building) when
#'     `sample = FALSE`.
#'   \item A JAX array of samples drawn from the RelaxedBernoulli distribution
#'     (for direct sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.relaxed_bernoulli(
#'     temperature = 1.0,
#'     probs = jnp$array(c(0.2, 0.3, 0.5)),
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli}
#'
#' @references
#' \[1\]:
#' https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedBernoulli
#' "tfp.distributions.RelaxedBernoulli | TensorFlow Probability"
#'
#' \[2\]:
#' https://rstudio.github.io/tfprobability/reference/tfd_relaxed_bernoulli.html
#' "RelaxedBernoulli distribution with temperature and logits ..."
#'
#' @export
bf.dist.relaxed_bernoulli=function(temperature, probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(logits)){logits=.BF_env$jnp$asarray(logits)}
     if(!.BF_env$.py$is_none(probs)){probs=.BF_env$jnp$asarray(probs)}
     .BF_env$.bf_instance$dist$relaxed_bernoulli(
       temperature = .BF_env$jnp$asarray(temperature),
       probs = probs,
       logits = logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

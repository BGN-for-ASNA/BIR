#' @title Relaxed Bernoulli Logits distribution
#'
#' @description
#' The Relaxed Bernoulli (logits) is a **continuous** relaxation of the standard
#' Bernoulli distribution, parameterised by logits (or probabilities) and a
#' **temperature** parameter. Rather than producing strictly 0 or 1, it produces
#' values in the continuous interval (0, 1). As the temperature → 0 the
#' distribution approximates a Bernoulli; as temperature → \eqn{\infty} the
#' distribution approximates a uniform distribution.
#'
#' It is used in variational inference and deep-learning contexts to allow
#' gradient-based optimisation through otherwise discrete Bernoulli draws (see
#' the “Concrete” or “Gumbel-Softmax” literature: The Concrete Distribution: A
#' Continuous Relaxation of Discrete Random Variables  and Categorical
#' Reparameterization with Gumbel‑Softmax ).
#'
#' Because this is a continuous relaxation, the “density” is defined over the
#' unit interval (0,1) rather than a pmf at \{0,1\}. Using logistic or
#' Gumbel‐softmax style construction one can express it as:
#'
#' If logits = ( \\ell ), and temperature = ( \\tau > 0 ), then one generates
#'
#' \deqn{u \sim \mathrm{Uniform}(0,1),\quad g = -\log(-\log(u)),\quad x = \sigma\big((\ell + g)/\tau\big)}
#'
#' where \eqn{\sigma(\cdot)} is the logistic (sigmoid) function. Then \eqn{x \in
#' (0,1)} has the RelaxedBernoulli distribution.
#'
#' The exact density formula can be derived via change of variables from
#' logistic/Gumbel, but is somewhat involved (including Jacobian of sigmoid
#' transform).
#'
#' @param temperature The temperature parameter, must be positive.
#' @param logits The logits parameter.
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
#'   \item RelaxedBernoulliLogits: A BF RelaxedBernoulliLogits distribution object
#'     (for model building).
#'   \item numeric vector or array: A JAX array of samples drawn from the
#'     RelaxedBernoulliLogits distribution (for direct sampling).
#'   \item RelaxedBernoulliLogits: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.relaxed_bernoulli_logits(temperature = 1.0, logits = 0.0, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#relaxed-bernoulli-logits}
#'
#' @references
#' https://arxiv.org/pdf/1611.00712
#'
#' @export
bf.dist.relaxed_bernoulli_logits=function(temperature, logits, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$relaxed_bernoulli_logits(
       temperature = .BF_env$jnp$asarray(temperature),
       logits = .BF_env$jnp$asarray(logits),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

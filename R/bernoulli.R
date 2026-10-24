#' @title Bernoulli distribution
#'
#' @description
#' The Bernoulli distribution models a single trial with two possible outcomes:
#' success or failureIt models a single trial with exactly two possible
#' outcomes:
#'
#' * **Success** (often coded as 1) with probability (p),
#' * **Failure** (often coded as 0) with probability (1 - p).
#'
#' It is a building block for many more complex discrete distributions (e.g.
#' binomial, geometric).
#'
#' We write
#'
#' \deqn{X \sim \mathrm{Bernoulli}(p)}
#'
#' with \eqn{0 \le p \le 1}.
#'
#' ## Probability Mass Function (PMF)
#'
#' \deqn{P(X = x) = \begin{cases} p, & \text{if } x = 1, \ 1 - p, & \text{if } x = 0, \ 0, & \text{otherwise.} \end{cases}}
#'
#' Alternatively, using an “indicator” exponent form:
#'
#' \deqn{P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in {0,1}.}
#'
#' @param probs Probability of success for each Bernoulli trial. Must be between
#'   0 and 1.
#' @param logits Log-odds of success for each Bernoulli trial.  `probs =
#'   sigmoid(logits)`.
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
#'   \item BF Bernoulli distribution object (for model building) when `sample =
#'     FALSE`.
#'   \item JAX array of samples drawn from the Bernoulli distribution (for direct
#'     sampling) when `sample = TRUE`.
#'   \item The raw BF distribution object (for advanced use cases) when
#'     `create_obj = TRUE`.
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.bernoulli(probs = 0.7, sample = TRUE)
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#bernoulli}
#'
#' @references
#' \[1\]: https://en.wikipedia.org/wiki/Bernoulli_distribution "Bernoulli
#' distribution"
#'
#' \[2\]:
#' https://stats.libretexts.org/Courses/Saint_Mary\%27s_College_Notre_Dame/MATH_345__-_Probability_\%28Kuter\%29/3\%3A_Discrete_Random_Variables/3.3\%3A_Bernoulli_and_Binomial_Distributions
#' "3.3: Bernoulli and Binomial Distributions - Statistics LibreTexts"
#'
#' @export
bf.dist.bernoulli=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if (.BF_env$.py$is_none(logits)){
      .BF_env$.bf_instance$dist$bernoulli(probs=.BF_env$jnp$asarray(probs),   validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
     }else{
       .BF_env$.bf_instance$dist$bernoulli(logits= .BF_env$jnp$asarray(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= .bf_infer(infer, enum),  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)

     }
}

#' @title Zero-Inflated Negative Binomial distribution
#'
#' @description
#' A Zero-Inflated Negative Binomial distribution is used for count data that
#' exhibit **both** (a) over-dispersion relative to a Poisson (i.e., variance >
#' mean) *and* (b) an excess of zero counts beyond what a standard Negative
#' Binomial would predict. It assumes two latent processes:
#'
#' 1. With probability \eqn{\pi} (sometimes denoted \eqn{\psi} or
#' “zero-inflation probability”) you are in a “structural zero” state → you
#' observe a zero.
#' 2. With probability \eqn{1 - \pi}, you come from a regular Negative Binomial
#' distribution (with parameters e.g. mean \eqn{\mu} and dispersion parameter
#' \eqn{\alpha} or size/r parameter) and then you might observe zero or a
#' positive count.
#'
#' Thus the model is a mixture of a point‐mass at zero + a Negative Binomial for
#' counts.
#'
#' This distribution combines a Negative Binomial distribution with a binary
#' gate variable. Observations are either drawn from the Negative Binomial
#' distribution with probability (1 - gate) or are treated as zero with
#' probability 'gate'. This models data with excess zeros compared to what a
#' standard Negative Binomial distribution would predict.
#'
#' Let \eqn{X} denote the count random variable, support \eqn{{0,1,2,\dots}}.
#' Denote:
#'
#' * \eqn{\pi} = probability of being in the “always zero” (structural zero)
#' process, with \eqn{0 \le \pi \le 1}.
#' * The count process is NB with parameters: mean \eqn{\mu>0} and
#' dispersion/shape parameter \eqn{\alpha>0} (or equivalently size parameter
#' (r)). Then:
#'
#' \deqn{P(X = 0) = \pi ;+; (1 - \pi); P_{\text{NB}}(0;\mu,\alpha),}
#'
#' \deqn{P(X = k) = (1 - \pi); P_{\text{NB}}(k;\mu,\alpha), \quad k = 1,2,3,\dots}
#'
#' Here \eqn{P_{\text{NB}}(k;\mu,\alpha)} is the PMF of the negative binomial
#' distribution for count \eqn{k}.
#'
#' @param mean The mean of the Negative Binomial 2 distribution.
#' @param concentration The concentration parameter of the Negative Binomial 2
#'   distribution.
#' @param gate See the Python documentation of `gate`.
#' @param gate_logits See the Python documentation of `gate_logits`.
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
#'   \item When `sample = FALSE`: A BF ZeroInflatedNegativeBinomial2 distribution
#'     object (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     ZeroInflatedNegativeBinomial2 distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.zero_inflated_negative_binomial(
#'     mean = 2.0,
#'     concentration = 3.0,
#'     gate = 0.3,
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#zeroinflatednegativebinomial2}
#'
#' @references
#' \[1\]: https://www.ewadirect.com/proceedings/tns/article/view/27624 "Social
#' Networks Count Data: Negative Binomial Distributions and Extensions Versus
#' Poisson and Bernoulli Models"
#'
#' \[2\]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7880198/ "Bayesian
#' Zero-Inflated Negative Binomial Regression Based on Pólya-Gamma Mixtures -
#' PMC"
#'
#' \[3\]: https://search.r-project.org/CRAN/refmans/emdbook/html/dzinbinom.html
#' "R: Zero-inflated negative binomial distribution"
#'
#' @export
bf.dist.zero_inflated_negative_binomial=function(mean, concentration, gate=py_none(), gate_logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE ) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}

     if(!.BF_env$.py$is_none(gate)){gate = .BF_env$jnp$asarray(gate)}
     if(!.BF_env$.py$is_none(gate_logits)){gate_logits = .BF_env$jnp$asarray(gate_logits)}


     .BF_env$.bf_instance$dist$zero_inflated_negative_binomial2(
       mean = .BF_env$jnp$asarray(mean),
       concentration = .BF_env$jnp$asarray(concentration),
       gate = gate,
       gate_logits = gate_logits,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @rdname bf.dist.zero_inflated_negative_binomial
#' @param ... For `bf.dist.zero_inflated_negative_binomial2()`, the arguments of
#'   `bf.dist.zero_inflated_negative_binomial()` (same function, Python name).
#' @export
bf.dist.zero_inflated_negative_binomial2 <- function(...) bf.dist.zero_inflated_negative_binomial(...)

#' @title Left Truncated distribution
#'
#' @description
#' Samples from a left-truncated distribution.
#'
#' A left-truncated distribution is a probability distribution obtained by
#' restricting the support of another distribution to values greater than a
#' specified lower bound. This is useful when dealing with data that is known to
#' be greater than a certain value. All the “mass” below (or equal to) (a) is
#' **excluded** (not just unobserved, but removed from the sample/analysis).
#'
#' Let (X) be a random variable with PDF (f_X(x)) and CDF (F_X(x)). Choose a
#' truncation point (a). Then define a new random variable (Y = X \\mid (X >
#' a)). The PDF and CDF of (Y) are:
#'
#' * **PDF** for ( y > a ):
#'
#' \deqn{f_Y(y) = \frac{f_X(y)}{1 - F_X(a)} .}
#'
#' (Here (1 - F_X(a) = P(X > a)) is the normalising constant)
#' (\[Wikipédia\]\[1\])
#'
#' * **CDF** for (y > a):
#'
#' \deqn{F_Y(y) = P(Y \le y) = \frac{F_X(y) - F_X(a)}{1 - F_X(a)} .}
#'
#' @param base_dist The base distribution to truncate. Must be univariate and
#'   have real support.
#' @param low The lower truncation bound. Values less than this are excluded
#'   from the distribution.
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
#'   \item When `sample = FALSE`: A BF LeftTruncatedDistribution distribution
#'     object (for model building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     LeftTruncatedDistribution distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF distribution object (for advanced
#'     use cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.left_truncated_distribution(
#'     base_dist = bf.dist.normal(0.0, 1.0, create_obj = TRUE),
#'     low = 0.0,
#'     sample = TRUE
#' )
#' }
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#lefttruncateddistribution}
#'
#' @references
#' https://en.wikipedia.org/wiki/Truncated_distribution
#'
#' https://encyclopediaofmath.org/wiki/Truncated_distribution
#'
#' @export
bf.dist.left_truncated_distribution=function(base_dist, low=0.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None")
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BF_env$.bf_instance$dist$left_truncated_distribution(
       base_dist,
       low = .bf_arr(low),
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

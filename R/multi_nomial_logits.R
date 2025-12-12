#' @title Multinomial logit
#' @description
#' A *multinomial logits* distribution refers to a categorical (or more generally multinomial)
#' distribution over \eqn{K} classes whose probabilities are given via the softmax of a vector of logits.
#' That is, given a vector of real-valued logits \eqn{\ell = (\ell_1, \dots, \ell_K)}, the class probabilities are:
#' \deqn{
#'   p_k = \frac{\exp(\ell_k)}{\sum_{j=1}^K \exp(\ell_j)}.
#'  }
#' Then a single draw from the distribution yields one of the \eqn{K} classes (or for a multinomial count version, counts over the classes) with those probabilities.#' @export
#' @importFrom reticulate py_none tuple
#'
#' @param logits A numeric vector, matrix, or array representing the logits for each outcome.
#' @param total_count A numeric vector, matrix, or array representing the total number of trials.
#' @param total_count_max (int, optional): An optional integer providing an upper bound on `total_count`. This is used for performance optimization with `lax.scan` when `total_count` is a dynamic JAX tracer, helping to avoid recompilation.

#' @param shape A numeric vector specifying the shape of the distribution.  Use a vector (e.g., `c(10)`) to define the shape.
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector, matrix, or array to mask observations.
#' @param create_obj Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#' @param to_jax Boolean. Indicates whether to return a JAX array or not.
#'
#' @return
#'    - When \code{sample=FALSE}, a BI MultinomialLogits distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MultinomialLogits distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#multinomiallogits}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.multinomial_logits(logits =  c(0.2, 0.3, 0.5), total_count = 10, sample = TRUE)
#' }
#' @export
bi.dist.multinomial_logits=function(logits, total_count=1, total_count_max=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     if(!.BI_env$.py$is_none(total_count_max)){total_count_max= as.integer(total_count_max)}
     total_count= .BI_env$jnp$array(as.integer(total_count))

     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$multinomial_logits(
       logits = .BI_env$jnp$array(logits),
       total_count = .BI_env$jnp$array(total_count),
       total_count_max = total_count_max,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

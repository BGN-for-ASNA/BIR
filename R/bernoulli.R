#' @title Sample from a Bernoulli distribution.
#'
#' @description The Bernoulli distribution models a single trial with two possible outcomes: success or failure.
#' It is parameterized by the probability of success, often denoted as 'p'.
#'
#' \deqn{P(X=1) = p \\ P(X=0) = 1 - p}
#'
#'
#' @param probs A numeric vector, matrix, or array representing the probability of success for each Bernoulli trial. Must be between 0 and 1.
#'
#' @param logits A numeric vector, matrix, or array representing the log-odds of success for each Bernoulli trial. `probs = sigmoid(logits)`.
#'
#' @param shape A numeric vector specifying the shape of the output.  Used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#'
#' @param event An integer indicating the number of batch dimensions to reinterpret as event dimensions (used in model building).
#'
#' @param mask A logical vector, matrix, or array (optional) to mask observations.
#'
#' @param create_obj A logical value (optional). If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
#'
#' @param validate_args Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
#'
#' @param sample A logical value that controls the function's behavior. If `TRUE`,
#'   the function will directly draw samples from the distribution. If `FALSE`,
#'   it will create a random variable within a model. Defaults to `FALSE`.
#'
#' @param seed An integer used to set the random seed for reproducibility when
#'   `sample = TRUE`. This argument has no effect when `sample = FALSE`, as
#'   randomness is handled by the model's inference engine. Defaults to 0.
#'
#' @param obs A numeric vector or array of observed values. If provided, the
#'   random variable is conditioned on these values. If `NULL`, the variable is
#'   treated as a latent (unobserved) variable. Defaults to `NULL`.
#'
#' @param name A character string representing the name of the random variable
#'   within a model. This is used to uniquely identify the variable. Defaults to 'x'.
#'
#' @return
#'  - When \code{sample=FALSE}, a BI Bernoulli distribution object (for model building).
#'
#'  - When \code{sample=TRUE}, a JAX array of samples drawn from the Bernoulli distribution (for direct sampling).
#'
#'  - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).

#' @examples
#' \donttest{
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.bernoulli(probs = 0.5, sample = TRUE)
#' bi.dist.bernoulli(probs = 0.5, sample = TRUE, seed = 5)
#' bi.dist.bernoulli(logits = 1, sample = TRUE, seed = 5)
#' }
#' @export
bi.dist.bernoulli=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     reticulate::py_run_string("def is_none(x): return x is None")
     if (.py$is_none(logits)){
      .bi$dist$bernoulli(probs=jnp$array(probs),   validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
     }else{
       .bi$dist$bernoulli(logits= jnp$array(logits),  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)

     }
}

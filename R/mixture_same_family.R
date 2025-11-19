#' @title A finite mixture of component distributions from the same family.
#' @description
#' A *Mixture (Same-Family)* distribution is a finite mixture in which **all components come from the same parametric family** (for example, all Normal distributions but with different parameters), and are combined via mixing weights. The class is typically denoted as:

#'  Let the mixture have (K) components. Let weights \deqn{w_i\ge0}, \deqn{\sum_{i=1}^K w_i = 1}. Let the component family have density \deqn{f(x \mid \theta_i)} for each component (i). Then the mixture's PDF is
#' \deqn{
#'  f_X(x) = \sum_{i=1}^K w_i ; f(x \mid \theta_i).
#' }
#' where each $f(x \mid \theta_i)$ is from the same family with parameter $\theta_i$.
#'
#' @param mixing_distribution A distribution specifying the weights for each mixture component.
#'   The size of this distribution specifies the number of components in the mixture.
#' @param component_distribution A list of distributions representing the components of the mixture.
#' @param shape A numeric vector specifying the shape of the distribution.
#' @param event Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
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
#'    - When \code{sample=FALSE}, a BI MixtureSameFamily distribution object (for model building).
#'
#'    - When \code{sample=TRUE}, a JAX array of samples drawn from the MixtureSameFamily distribution (for direct sampling).
#'
#'    - When \code{create_obj=TRUE}, the raw BI distribution object (for advanced use cases).
#'
#' @seealso This is a wrapper of  \url{https://num.pyro.ai/en/stable/distributions.html#mixture-same-family}
#'
#' @examples
#' \donttest{
#' library(BayesianInference)
#' m=importBI(platform='cpu')
#' bi.dist.mixture_same_family(
#' mixing_distribution = bi.dist.categorical(probs = c(0.3, 0.7),create_obj = TRUE),
#' component_distribution = bi.dist.normal(0,1, shape = c(2), create_obj = TRUE),
#' sample = TRUE)
#' }
#' @export
bi.dist.mixture_same_family=function(mixing_distribution, component_distribution, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     reticulate::py_run_string("def is_none(x): return x is None");
     if (!.BI_env$.py$is_none(seed)){seed=as.integer(seed);}
     .BI_env$.bi_instance$dist$mixture_same_family(
       mixing_distribution,
       component_distribution ,
       validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

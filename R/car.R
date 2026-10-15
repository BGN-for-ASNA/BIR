#' @title Conditional Autoregressive (CAR) distribution
#'
#' @description
#' The CAR distribution models a vector of variables where each variable is a
#' linear combination of its neighbors in a graph. The CAR model captures
#' spatial dependence in areal data by modeling each observation as
#' conditionally dependent on its neighbors.It specifies a joint distribution of
#' a vector of random variables \eqn{\mathbf{y} = (y_1, y_2, \dots, y_N)} based
#' on their conditional distributions, where each \eqn{y_i} is conditionally
#' independent of all other variables given its neighbors.
#'
#' * **Application**: Widely used in disease mapping, environmental modeling,
#' and spatial econometrics to account for spatial autocorrelation.
#'
#' * **Conditional Distribution**: Each \eqn{y_i} given its neighbors
#' \eqn{\mathbf{y}_{-i}} follows a normal distribution:
#'
#' \deqn{y_i | \mathbf{y}*{-i} \sim \mathcal{N}\left( \frac{\sum*{j \in \mathcal{N}(i)} w_{ij} y_j}{\sum_{j \in \mathcal{N}(i)} w_{ij}}, \sigma^2 \right)}
#'
#' where:
#'
#' * \eqn{\mathcal{N}(i)} denotes the set of neighbors of unit \eqn{i},
#' * \eqn{w_{ij}} are the weights representing the strength of the spatial
#' relationship between units \eqn{i} and \eqn{j},
#' * \eqn{\sigma^2} is the variance term. where \eqn{\mu_i} is a function of the
#' values of the neighbors of site \eqn{i} and \eqn{\Sigma_i} is the variance of
#' site \eqn{i}.
#'
#' @param loc Mean of the distribution.
#' @param correlation Correlation between variables.
#' @param conditional_precision Precision of the distribution.
#' @param adj_matrix Adjacency matrix defining the graph. is_sparse (bool):
#'   Whether the adjacency matrix is sparse. Defaults to FALSE. - *validate_args/
#'   (bool): Whether to validate arguments. Defaults to py_none().
#' @param is_sparse See the Python documentation of `is_sparse`.
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
#'   \item When `sample = FALSE`: A BF CAR distribution object (for  model
#'     building).
#'   \item When `sample = TRUE`: A JAX array of samples drawn from the
#'     BernoulliLogits distribution (for direct sampling).
#'   \item When `create_obj = TRUE`: The raw BF CAR object (for advanced use
#'     cases).
#' }
#'
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' bf.dist.car(loc = 0.0, correlation = 0.9, conditional_precision = 1.0, adj_matrix = jnp$array(matrix(c(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0), nrow = 3L, byrow = TRUE)), sample = TRUE)
#' }
#'
#' @references
#' \[1\]:
#' https://www.pymc.io/projects/examples/en/latest/spatial/conditional_autoregressive_priors.html
#' "Conditional Autoregressive (CAR) Models for Spatial Data"
#'
#' \[2\]: https://mc-stan.org/learn-stan/case-studies/icar_stan.html "Intrinsic
#' Auto-Regressive Models for Areal Data - Stan"
#'
#' \[3\]:
#' https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118445112.stat08048
#' "Conditional Autoregressive (CAR) Model - Schmidt"
#'
#' @export
bf.dist.car=function(loc, correlation, conditional_precision, adj_matrix, is_sparse=FALSE, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed = py_none(), shape=c(), event=0, create_obj=FALSE, to_jax = TRUE) {
     if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))  # obs = y -> site "y"

     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     if (!.BF_env$.py$is_none(seed)){seed=as.integer(seed);}
     if(!.BF_env$.py$is_none(correlation)){correlation = .BF_env$jnp$asarray(correlation)}
     if(!.BF_env$.py$is_none(conditional_precision)){conditional_precision = .BF_env$jnp$asarray(conditional_precision)}
     if(!.BF_env$.py$is_none(adj_matrix)){adj_matrix = .BF_env$jnp$asarray(adj_matrix)}
     .BF_env$.bf_instance$dist$car(
       loc = .BF_env$jnp$asarray(loc),
       correlation=correlation,
       conditional_precision=conditional_precision,
       adj_matrix=adj_matrix,
       is_sparse= is_sparse,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,   to_jax = to_jax)
}

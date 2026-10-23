#' Posterior / prior predictions as R arrays
#'
#' Calls `m$sample()` and converts the returned JAX arrays into plain R objects,
#' so the result can be used with any R function (`colMeans()`, `apply()`,
#' matrix arithmetic, ...).
#'
#' `m$sample()` returns a named list whose values are JAX arrays (Python
#' objects); R functions cannot work with those. `bf_sample()` keeps the names
#' and turns every value into an R array: a `(draws, observations)` prediction
#' becomes an R matrix, a 1-D result stays a plain numeric vector.
#'
#' The arrays returned by `m$sample()` itself are left untouched -- model code
#' must keep receiving JAX arrays. Use [as.array()], [as.matrix()] or
#' [as.vector()] on a single JAX array when only that one is needed.
#'
#' @param m A BayesForge object (from [importBF()]).
#' @param ... Passed on to `m$sample()`, e.g. `data =`, `samples =`, `seed =`,
#'   `remove_obs =`.
#'
#' @return A named list of R arrays (or a single R array when `m$sample()`
#'   returns one array instead of a list).
#' @examples
#' \dontrun{
#' m <- importBF()
#' # ... m$fit(model) ...
#' pp <- bf_sample(m)
#' dim(pp$height)      # 4000 x 346
#' colMeans(pp$height)
#' }
#' @export
bf_sample <- function(m, ...) {
  res <- m$sample(...)
  if (is.list(res)) {
    return(lapply(res, .bf_jax_to_r))
  }
  if (!inherits(res, "python.builtin.dict")) {
    # a single array rather than a named collection
    return(.bf_jax_to_r(res))
  }
  nms <- names(res)
  out <- lapply(nms, function(nm) .bf_jax_to_r(res[[nm]]))
  names(out) <- nms
  out
}

# A JAX array (or any other Python object) -> the equivalent R array.
# Values that are already R objects are returned unchanged.
.bf_jax_to_r <- function(x) {
  if (inherits(x, "python.builtin.object")) {
    np <- reticulate::import("numpy", convert = FALSE)
    return(reticulate::py_to_r(np$asarray(x)))
  }
  x
}

# as.array() / as.matrix() / as.vector() on a JAX array.
# Registered in NAMESPACE as S3method(as.array, jax.Array, .bf_jax_as_array), ...
.bf_jax_as_array <- function(x, ...) {
  a <- .bf_jax_to_r(x)
  if (is.null(dim(a))) array(a, dim = length(a)) else a
}

.bf_jax_as_matrix <- function(x, ...) as.matrix(.bf_jax_to_r(x), ...)

.bf_jax_as_vector <- function(x, mode = "any") as.vector(.bf_jax_to_r(x), mode = mode)

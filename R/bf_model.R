#' Write a BayesForge model with automatic parameter names
#'
#' Wraps an R model function so that every parameter assigned from a
#' distribution gets the name of its variable, as in Python and in Julia
#' (`@BF`): `a <- bf.dist.normal(0, 1)` becomes
#' `a <- bf.dist.normal(0, 1, name = "a")`. An explicit `name =` is kept.
#' Observed sites (`obs = y`) are already named after their variable by the
#' distribution functions.
#'
#' The model body is rewritten once, when `bf_model()` is called; the
#' function is otherwise unchanged. Both `<-` and `=` assignments are
#' handled, also inside `with(m$dist$plate(...), ...)` blocks. The
#' distribution calls recognised are `bf.dist.*(...)` and `m$dist$*(...)`.
#'
#' @param f A model function.
#' @return The same function, with `name =` added to its distribution calls.
#' @examples
#' \donttest{
#' library(BayesForge)
#' m <- importBF(platform = "cpu")
#' model <- bf_model(function(x, y) {
#'   a <- bf.dist.normal(0, 1)        # name "a"
#'   b <- bf.dist.normal(0, 1)        # name "b"
#'   s <- bf.dist.exponential(1)      # name "s"
#'   bf.dist.normal(a + b * x, s, obs = y)
#' })
#' }
#' @export
bf_model <- function(f) {
  if (!is.function(f)) stop("bf_model() needs a function.")
  body(f) <- .bf_add_names(body(f))
  f
}

# TRUE for bf.dist.normal(...) and for m$dist$normal(...).
.bf_is_dist_call <- function(e) {
  if (!is.call(e)) return(FALSE)
  fn <- e[[1]]
  if (is.name(fn)) return(startsWith(as.character(fn), "bf.dist."))
  is.call(fn) && identical(fn[[1]], as.name("$")) &&
    is.call(fn[[2]]) && identical(fn[[2]][[1]], as.name("$")) &&
    identical(fn[[2]][[3]], as.name("dist"))
}

.bf_add_names <- function(e) {
  if (!is.call(e)) return(e)
  is_assign <- identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))
  if (is_assign && length(e) == 3 && is.name(e[[2]]) &&
      .bf_is_dist_call(e[[3]]) && is.null(e[[3]][["name"]])) {
    e[[3]][["name"]] <- as.character(e[[2]])
  }
  for (i in seq_along(e)) {
    # e[[i]] is not copied to a variable first: an empty argument (x[, 1])
    # is the empty symbol, and a variable holding it counts as missing.
    if (is.call(e[[i]])) e[[i]] <- .bf_add_names(e[[i]])
  }
  e
}

#' Package load hook
#'
#' Internal function to run when the package is loaded.
#' @keywords internal
.BF_env <- new.env(parent = emptyenv())

# Define dummy fallback methods for non-interactive / uninitialized sessions
#' @export
`$.dummy_bf_instance` <- function(x, name) {
  if (name == "dist") {
    dummy_dist <- list()
    class(dummy_dist) <- "dummy_dist"
    return(dummy_dist)
  }
  return(function(...) {
    message("Note: Using dummy BFR function '", name, "' because Python is not loaded.")
    return(invisible(NULL))
  })
}

#' @export
`$.dummy_dist` <- function(x, name) {
  return(function(...) {
    message("Note: Using dummy BFR function '", name, "' because Python is not loaded.")
    return(invisible(NULL))
  })
}

.dummy_bf_instance <- list()
class(.dummy_bf_instance) <- "dummy_bf_instance"

.BF_env$loaded <- FALSE
.BF_env$.bf <- NULL
.BF_env$jnp <- list(array = function(x, ...) x)
.BF_env$.py <- list(is_none = function(x) is.null(x) || inherits(x, "python.builtin.NoneType"))
.BF_env$.bf_instance <- .dummy_bf_instance

#' @title Package Load
#' @description
#' Internal function to run when the package is loaded.
#' @param libname Library name.
#' @param pkgname Package name.
#' @return No return value, called for side effects.
#' @keywords internal
.onLoad <- function(libname = NULL, pkgname = "BayesForge") {
  # `loo` is only Suggested, so `loo.BayesForge.Main.main.bf` cannot be
  # declared in NAMESPACE (loading would fail when `loo` is absent).
  # Register it here when available.
  if (requireNamespace("loo", quietly = TRUE)) {
    registerS3method(
      "loo", "BayesForge.Main.main.bf",
      loo.BayesForge.Main.main.bf,
      envir = asNamespace("loo")
    )
  }
}

#' @title Package attach
#' @description
#' Internal function to run when the package is attached.
#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  packageStartupMessage("For documentation, run the command: bf.doc()")
  # This function runs ONLY in an interactive session to show a message.
  # packageStartupMessage("For documentation, run the command: bf.doc()")
  BI_venv_present <- check_env()
  if (BI_venv_present) {
    reticulate::use_virtualenv("BFR", required = TRUE)
  } else {
    packageStartupMessage("It seems that the BFR virtual environment is not set up. Please run: BF_starting_test()")
  }
}

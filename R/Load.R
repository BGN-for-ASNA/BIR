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
.BF_env$jnp <- list(array = function(x, ...) x, asarray = function(x, ...) x)
.BF_env$.py <- list(is_none = function(x) is.null(x) || inherits(x, "python.builtin.NoneType"))
.BF_env$.bf_instance <- .dummy_bf_instance

#' JAX modules
#'
#' `jnp` (`jax.numpy`) and `jax`, imported lazily from the BayesForge Python
#' environment, for building arrays passed to BayesForge models, e.g.
#' `jnp$array(c(1, 2, 3))` or `jnp$eye(3L)`.
#'
#' @name jax_modules
#' @aliases jnp jax
#' @export jnp
#' @export jax
NULL
jnp <- NULL
jax <- NULL

#' @title Package load hook
#' @description
#' Binds the session to the BayesForge Python environment when it exists
#' (before any Python is initialised) and sets up the lazy `jnp`/`jax`
#' modules.
#' @param libname Library name.
#' @param pkgname Package name.
#' @return No return value, called for side effects.
#' @keywords internal
.onLoad <- function(libname, pkgname) {
  if (reticulate::virtualenv_exists("BFR") && !reticulate::py_available(initialize = FALSE)) {
    reticulate::use_virtualenv("BFR", required = FALSE)
  }
  jnp <<- reticulate::import("jax.numpy", delay_load = TRUE)
  jax <<- reticulate::import("jax", delay_load = TRUE)
  # posterior is only suggested: register as_draws_df() for fitted models
  # when it is installed.
  if (requireNamespace("posterior", quietly = TRUE)) {
    registerS3method("as_draws_df", "BayesForge.Main.main.bf",
                     .as_draws_df_bf,
                     envir = asNamespace("posterior"))
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
  if (!reticulate::virtualenv_exists("BFR") && !nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
    packageStartupMessage(
      "The BayesForge Python environment is not set up yet. ",
      "Run setup_env() once (see ?setup_env)."
    )
  }
}

#' Package load hook
#'
#' Internal function to run when the package is loaded.
#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.BI_env <- new.env(parent = emptyenv())

onLoad <- function(libname = NULL, pkgname = "BI") {
  BI_starting_test()
}

#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  # This function runs ONLY in an interactive session to show a message.
  packageStartupMessage("For documentation, run the command: bi.doc()")
  BI_starting_test()

}


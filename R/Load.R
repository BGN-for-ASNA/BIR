#' Package load hook
#'
#' Internal function to run when the package is loaded.
#' @keywords internal
.BI_env <- new.env(parent = emptyenv())

#' @title Package Load
#' @description
#' Internal function to run when the package is loaded.
#' @param libname Library name.
#' @param pkgname Package name.
#' @keywords internal
onLoad <- function(libname = NULL, pkgname = "BI") {
  packageStartupMessage("For documentation, run the command: bi.doc()")
}

#' @title Package attach
#' @description
#' Internal function to run when the package is attached.
#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  packageStartupMessage("For documentation, run the command: bi.doc()")
  .BI_env <- new.env(parent = emptyenv())
  # This function runs ONLY in an interactive session to show a message.
  # packageStartupMessage("For documentation, run the command: bi.doc()")
  BI_venv_present <- check_env()
  if (BI_venv_present) {
    reticulate::use_virtualenv("BayesInference", required = TRUE)
  } else {
    packageStartupMessage("It seems that the BayesInference virtual environment is not set up. Please run: BI_starting_test()")
  }
}

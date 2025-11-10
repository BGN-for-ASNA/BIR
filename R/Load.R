#' Package load hook
#'
#' Internal function to run when the package is loaded.
#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.BI_env <- new.env(parent = emptyenv())

onLoad <- function(libname = NULL, pkgname = "BI") {
  packageStartupMessage("For documentation, run the command: bi.doc()")
}

#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  .BI_env <- new.env(parent = emptyenv())
  # This function runs ONLY in an interactive session to show a message.
  #packageStartupMessage("For documentation, run the command: bi.doc()")
  BI_venv_present = check_env()
  if(BI_venv_present){
    reticulate::use_virtualenv("BayesInference", required = TRUE)
  }else{
    packageStartupMessage("It seems that the BayesInference virtual environment is not set up. Please run: BI_starting_test()")
  }
}



#' Package load hook
#'
#' Internal function to run when the package is loaded.
#' @param libname Internal.
#' @param pkgname Internal.
#' @keywords internal
onLoad <- function(libname = NULL, pkgname = "BI") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("The 'reticulate' package is required but not installed. Please install it via install.packages('reticulate').")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    packageStartupMessage("Python is not available on this system. Please install Python before using this package.")
  }

  test3 <- tryCatch(
    {
      reticulate::py_run_string("import BI")
      message("Python module 'BI' is available.")
      TRUE   # just the last expression, no return()
    },
    error = function(e) {
      message("'BI' not found, install BayesInference")
      message('Use: reticulate::py_install("BayesInference", pip = TRUE)')
      FALSE  # note capital FALSE
    }
  )
  
  test1 = requireNamespace("reticulate", quietly = TRUE)
  test2 = reticulate::py_available(initialize = TRUE)

  if(any(!test1, !test2, !test3)){
    .BI_env$ready <- FALSE
  }else{.BI_env$ready <- TRUE}
}

#' @keywords internal
.onAttach <- function(libname, pkgname) {
  # This function runs ONLY in an interactive session to show a message.
  packageStartupMessage("For documentation, run the command: bi.doc()")
  test1 = requireNamespace("reticulate", quietly = TRUE)
  test2 = reticulate::py_available(initialize = TRUE)
  test3 = reticulate::py_module_available("BI")
  if(any(!test1, !test2, !test3)){
    .BI_env$ready <- FALSE
  }else{.BI_env$ready <- TRUE}
}

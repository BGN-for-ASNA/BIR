#' @keywords internal
onLoad <- function(libname = NULL, pkgname = "BI") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("The 'reticulate' package is required but not installed. Please install it via install.packages('reticulate').")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    packageStartupMessage("Python is not available on this system. Please install Python before using this package.")
  }

  if (!reticulate::py_module_available("BI")) {
    packageStartupMessage("Python package 'BI' not found; installing now...")
    # Consider making this installation optional or asking the user for confirmation.
    tryCatch({
      reticulate::py_install("BayesInference", pip = TRUE)
    }, error = function(e) {
      packageStartupMessage("Failed to install 'BayesInference'. Please install it manually using 'reticulate::py_install(\"BayesInference\", pip = TRUE)'.")
    })
  }
  test1 = requireNamespace("reticulate", quietly = TRUE)
  test2 = reticulate::py_available(initialize = TRUE)
  test3 = reticulate::py_module_available("BI")
  if(any(!test1, !test2, !test3)){
    .BI_env$ready <- FALSE
  }
}

#' @keywords internal
.onAttach <- function(libname, pkgname) {
  # This function runs ONLY in an interactive session to show a message.
  packageStartupMessage("For documentation, run the command: bi.doc()")
}

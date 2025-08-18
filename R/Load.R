.onAttach <- function(libname = NULL, pkgname = "BI") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("The 'reticulate' package is required but not installed. Please install it via install.packages('reticulate').")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    packageStartupMessage("Python is not available on this system. Please install Python before using this package.")
  }
  packageStartupMessage("For documentation run command :  bi.doc()")

  if (!reticulate::py_module_available("BI")) {
    packageStartupMessage("Python package 'BI' not found; installing now...")
    # Consider making this installation optional or asking the user for confirmation.
    tryCatch({
      reticulate::py_install("BayesInference", pip = TRUE)
    }, error = function(e) {
      packageStartupMessage("Failed to install 'BayesInference'. Please install it manually using 'reticulate::py_install(\"BayesInference\", pip = TRUE)'.")
    })
  }
}

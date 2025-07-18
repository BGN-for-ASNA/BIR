.pkg_env <- new.env(parent = emptyenv())

.onAttach <- function(libname=NULL, pkgname="BI") {
  require(reticulate)
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but not installed. Please install it via install.packages('reticulate').")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    stop("Python is not available on this system. Please install Python before using this package.")
  }
  packageStartupMessage("For documentation run command :  bi.doc()")

  if (!reticulate::py_module_available("BI")) {
    message("Python package 'BI' not found; installing now...")
    reticulate::py_install("BayesInference", pip = TRUE)
  }


}

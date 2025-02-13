.onAttach <- function(libname=NULL, pkgname="BI") {

  packageStartupMessage("For documentation run command :  bi.doc()")
  if (!reticulate::py_module_available("numpyro")) {
    # If numpyro is not available, install it.
    # You can install it via pip (default) or conda, depending on your needs.
    # The following installs using pip:
    packageStartupMessage("Python package 'numpyro' not found; installing now...")
    reticulate::py_install("numpyro", pip = TRUE)
  }
  if (!reticulate::py_module_available("jax")) {
    # If numpyro is not available, install it.
    # You can install it via pip (default) or conda, depending on your needs.
    # The following installs using pip:
    packageStartupMessage("Python package 'jax' not found; installing now...")
    reticulate::py_install("jax", pip = TRUE)
  }


  #if (!reticulate::py_module_available("BI")) {
  #  # If numpyro is not available, install it.
  #  # You can install it via pip (default) or conda, depending on your needs.
  #  # The following installs using pip:
  #  message("Python package 'BI' not found; installing now...")
  #  reticulate::py_install("BI", pip = TRUE)
  #}

}


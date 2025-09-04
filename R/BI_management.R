#' @title title  Install BayesInference Python Dependencies
#'
#' @description
#' Creates a dedicated Python virtual environment and installs the
#' `BayesInference` Python package and its required dependencies (like `numpyro`
#' and `jax`). It is essential to run this function once before using the package.
#'
#' @details
#' This function automates the setup process by:
#' 1. Creating a clean, isolated Python virtual environment named
#'    `r-bayes-inference-cpu` or `r-bayes-inference-gpu`.
#' 2. Installing the correct version of the `BayesInference` Python package
#'    from PyPI using `pip`.
#'
#' You must restart your R session after installation for the changes to take
#' effect.
#'
#' @param type (character) The installation type. Must be either `'cpu'` (default)
#'   for a CPU-only installation or `'gpu'` for a CUDA-enabled GPU installation.
#'
#' @return
#' Invisibly returns `NULL`. The function is called for its side effects.
#'
#' @keywords internal
BI_install <- function(type = "cpu") {

  # 4. Construct the correct package name with the extra
  package_to_install <- paste0("BayesInference[", type, "]")

  message("Installing ", package_to_install, "'...")

  # 5. Install the packages
  reticulate::py_require(
    packages = package_to_install,
    #envname = full_envname,
    pip = TRUE,
    pip_options = "--upgrade" # To ensure latest packages
  )

  message("\n----------------------------------------------------")
  message("Installation complete!")
  message("----------------------------------------------------")
}


#' @title Uninstall BayesInference Python Environment
#'
#' @description
#' Completely removes the Python virtual environment associated with the
#' BayesInference package.
#'
#' @details
#' This function is for cleanup and will delete the entire virtual environment,
#' including all installed Python packages. This action cannot be undone.
#'
#' @param type (character) The environment type to remove. Must be either
#'   `'cpu'` or `'gpu'`.
#'
#' @return
#' Invisibly returns `NULL`. The function is called for its side effects.
#'
#' @keywords internal
BI_uninstall <- function(){
  reticulate::virtualenv_remove(
    packages = "BayesInference"
  )
  packageStartupMessage("\n----------------------------------------------------")
  packageStartupMessage("BI uninstalled")
  packageStartupMessage("----------------------------------------------------")
}

#' @title Check if BayesInference Python package is installed
#' @description
#' Internal function to check whether the Python package `BayesInference`
#' is installed in the default reticualte virtual environment.
#'
#' @return Logical `TRUE` if the package is found, otherwise prints a message
#'   and returns `NULL`.
#' @keywords internal
BI_check_presence <- function(){
  r = reticulate::py_list_packages()
  test = "BayesInference" %in% r$package
  if(test){return(test)}else{
    packageStartupMessage("\n----------------------------------------------------")
    packageStartupMessage("BI could not be found!")
    packageStartupMessage("You need to isntall BI python version first with command : BI_install()")
    packageStartupMessage("If you already ran BI_install() and still see this message, you may need to force dependency versions using: BI_force_dependencies_version()")
    packageStartupMessage("----------------------------------------------------")
    return(FALSE)
  }
}

#' @title Force Installation of Specific Python Dependency Versions
#' @description
#' Internal function to install specific versions of Python packages required
#' for the package to function correctly. This ensures a consistent and stable
#' environment, avoiding potential conflicts from dependency updates.
#'
#' @details
#' This function uses `reticulate::py_install` to force the installation of
#' specific versions of `jax`, `jaxlib`, `numpyro`, `numpy`,
#' `tensorflow_probability`, `arviz`, `matplotlib`, `seaborn`, `pandas`, and
#' `scipy`. It is particularly useful for ensuring reproducibility of analyses.
#'
#' @return Invisible `NULL`. The function is called for its side effect of
#'   installing Python packages.
#' @keywords internal
#'
BI_force_dependencies_version <- function(){
  reticulate::py_require(
    packages = 'jax==0.5.1',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'jaxlib==0.5.1',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )


  reticulate::py_require(
    packages = 'numpyro== 0.18.0',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'numpy== 1.26.3',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )


  reticulate::py_require(
    packages = 'tensorflow_probability== 0.24.0',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'arviz==0.17.0',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'matplotlib==3.8.2',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'seaborn==0.13.2',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )


  reticulate::py_require(
    packages = 'pandas==2.2.3',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )

  reticulate::py_require(
    packages = 'scipy  == 1.11.4',
    #envname = full_envname,
    pip = TRUE,
    pip_options = c('--upgrade')
  )
}

#' @title Load the BI module from Python
#' @details
#' Internal helper function to import the `BI` Python module.
#'
#' @return The Python object corresponding to `BI$bi`.
#' @keywords internal
BI_load <- function(){
  tryCatch({reticulate::import("BI")$bi}, error = function(e) {
    message("\n----------------------------------------------------")
    message("An error occurred: ", e$message)
    message("----------------------------------------------------")
  })
}


#' @title Run a starting test for the BI environment
#' @details
#' Internal function that checks whether Python and the `BayesInference`
#' package are available, and sets internal `.BI_env` flags accordingly.
#'
#' @details
#' Checks if the `reticulate` package is installed, if Python is available,
#' and if the `BayesInference` package exists in the target environment.
#' Updates `.BI_env$loaded` and `.BI_env$.bi` accordingly.
#'
#' @keywords internal
BI_starting_test <- function(){
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("The 'reticulate' package is required but not installed. Please install it via install.packages('reticulate').")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    packageStartupMessage("Python is not available on this system. Please install Python before using this package.")
  }

  return = BI_check_presence()

}

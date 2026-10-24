#' @title Ask Yes/No Question
#'
#' @description
#' This function prompts the user to answer a yes/no question.
#'
#' @param prompt A character string specifying the question to ask.
#' @return A character string "Y" or "N" based on the user's response.
#' @keywords internal
ask_yes_no <- function(prompt = "Do you want to continue? (Y/N): ") {
  if (!interactive()) {
    return("N")
  }
  repeat {
    ans <- toupper(trimws(readline(prompt)))
    if (ans %in% c("Y", "N")) {
      return(ans)
    } else {
      cat("Please enter Y or N.\n")
    }
  }
}

# Minimum Python version required by the 'bayesforge' Python package.
.bf_python_version <- ">=3.12"
# Minimum 'bayesforge' (Python) version this R package works with. Both values
# must agree with compat.toml of BGN-for-ASNA/BF (checked in CI).
.bf_bayesforge_version <- ">=0.0.59"

#' Check if the default virtual environment is available
#'
#' This function checks for the existence of the default Python virtual environments.
#'
#' @param env_name Name of the virtual environment. Defaults to "BFR".
#' @param verbose Print whether the environment was found.
#' @return A logical value indicating whether the environment exists.
#' @keywords internal
check_env <- function(env_name = "BFR", verbose = TRUE) {
  exists <- reticulate::virtualenv_exists(envname = env_name)
  if (verbose) {
    packageStartupMessage(
      "Virtual environment ('", env_name, "') is ",
      if (exists) "available." else "not available."
    )
  }
  exists
}

#' Find a Python interpreter able to host BayesForge
#'
#' @return Path to a Python >= 3.12 interpreter.
#' @keywords internal
find_python <- function() {
  python <- reticulate::virtualenv_starter(.bf_python_version)
  if (!is.null(python)) {
    return(python)
  }
  if (ask_yes_no(paste0(
    "BayesForge needs Python ", .bf_python_version,
    " and none was found. Install Python 3.12 now? (Y/N): "
  )) == "Y") {
    return(reticulate::install_python(version = "3.12:latest"))
  }
  stop(
    "BayesForge needs Python ", .bf_python_version, ". Install it with ",
    "reticulate::install_python(\"3.12:latest\") and run setup_env() again.",
    call. = FALSE
  )
}

#' Create (or upgrade) the Python environment used by BayesForge
#'
#' Creates a dedicated Python virtual environment (Python >= 3.12) and installs
#' the `bayesforge` Python package into it, with JAX for the chosen backend.
#' Versions of JAX, NumPyro and ArviZ are resolved by pip from the requirements
#' of `bayesforge` itself, so the environment always matches the installed
#' BayesForge release. Running it again upgrades an existing environment.
#'
#' @param env_name Name of the virtual environment. Defaults to "BFR".
#' @param backend Either "cpu" or "gpu" (Linux/WSL2 with an NVIDIA GPU).
#' @param python Optional path to the Python interpreter used to create the
#'   environment. By default a Python >= 3.12 is located automatically.
#'
#' @return The path to the Python interpreter of the environment (invisibly).
#' @examples
#' \dontrun{
#' setup_env()                # CPU
#' setup_env(backend = "gpu") # NVIDIA GPU, Linux/WSL2 only
#' }
#' @export
setup_env <- function(env_name = "BFR", backend = c("cpu", "gpu"), python = NULL) {
  backend <- match.arg(backend)
  if (!reticulate::virtualenv_exists(envname = env_name)) {
    packageStartupMessage("Creating virtual environment: ", env_name)
    if (is.null(python)) python <- find_python()
    reticulate::virtualenv_create(envname = env_name, python = python)
  }
  pkg <- sprintf("bayesforge[%s]%s", backend, .bf_bayesforge_version)
  packageStartupMessage("Installing ", pkg, " into '", env_name, "' (this can take a few minutes)...")
  reticulate::virtualenv_install(envname = env_name, packages = pkg)
  packageStartupMessage("BayesForge Python environment ready.")
  invisible(reticulate::virtualenv_python(envname = env_name))
}

#' Bind this R session to the BayesForge Python environment
#'
#' @param env_name Name of the virtual environment. Defaults to "BFR".
#' @keywords internal
use_bf_env <- function(env_name = "BFR") {
  if (!reticulate::py_available(initialize = FALSE)) {
    reticulate::use_virtualenv(env_name, required = TRUE)
    return(invisible(TRUE))
  }
  # Python is already initialised in this session: it can no longer be
  # switched, so make sure it is the BayesForge environment.
  current <- normalizePath(reticulate::py_config()$python, mustWork = FALSE)
  expected <- normalizePath(reticulate::virtualenv_python(env_name), mustWork = FALSE)
  if (!identical(current, expected)) {
    stop(
      "This R session is already using the Python at ", current,
      ", not the BayesForge environment ('", env_name, "'). Restart R and run ",
      "library(BayesForge) before any other code that uses reticulate.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' @title Make sure the BayesForge Python environment is ready
#' @details
#' Internal function called by [importBF()]. If the "BFR" environment exists,
#' the session is bound to it. If not, but the user selected their own Python
#' (`RETICULATE_PYTHON`, `reticulate::use_python()`, ...) and `BayesForge` is
#' installed there, that Python is used. Otherwise the user is offered to
#' create the environment with [setup_env()] (interactive sessions), or an
#' error explains how to do so. It never reinstalls an existing environment.
#'
#' @keywords internal
BF_starting_test <- function() {
  user_python <- nzchar(Sys.getenv("RETICULATE_PYTHON")) ||
    reticulate::py_available(initialize = FALSE)
  if (!check_env(verbose = FALSE) && user_python &&
      reticulate::py_module_available("BayesForge")) {
    packageStartupMessage("Using BayesForge from ", reticulate::py_config()$python)
    return(invisible(TRUE))
  }
  if (!check_env(verbose = FALSE)) {
    if (ask_yes_no(paste0(
      "The BayesForge Python environment ('BFR') is not set up. ",
      "Create it now? (Y/N): "
    )) != "Y") {
      stop("The BayesForge Python environment is not set up. Run setup_env() once.", call. = FALSE)
    }
    gpu <- ask_yes_no("Use the GPU backend (Linux/WSL2 with an NVIDIA GPU only)? (Y/N): ") == "Y"
    setup_env(backend = if (gpu) "gpu" else "cpu")
  }
  use_bf_env()
}

#' @title Remove a virtual environment
#' @description
#' Internal helper function to remove a virtual environment.
#' @param env_name A character string specifying the name of the virtual environment.
#' @keywords internal
remove_env_name <- function(env_name = "BFR") {
  # Check if the virtual environment exists
  if (reticulate::virtualenv_exists(envname = env_name)) {
    message(paste("Found virtual environment:", env_name))

    # Remove the environment, with an interactive confirmation prompt
    reticulate::virtualenv_remove(envname = env_name, confirm = FALSE)

    # Optional: A final check to confirm it's gone
    if (!reticulate::virtualenv_exists(envname = env_name)) {
      message(paste("Successfully removed virtual environment:", env_name))
    }
  } else {
    message(paste("Virtual environment", env_name, "not found. Nothing to remove."))
  }
}


#' @title Load the BayesForge module from Python
#' @details
#' Internal helper function to import the `BayesForge` Python module.
#'
#' @return The Python object corresponding to `BayesForge$bf`.
#' @keywords internal
BF_load <- function() {
  reticulate::import("BayesForge")$bf
}

#' @title Update the BayesForge Python package
#' @details
#' Upgrades `bayesforge` (and, through its requirements, JAX/NumPyro/ArviZ)
#' in the environment.
#' @param envname A character string specifying the name of the virtual environment.
#' @param backend Either "cpu" or "gpu".
#' @keywords internal
update_BF <- function(envname = "BFR", backend = "cpu") {
  setup_env(env_name = envname, backend = backend)
}

#' @title List packages in a virtual environment
#' @details
#' Internal helper function to list packages in a virtual environment.
#' @param envname A character string specifying the name of the virtual environment.
#' @keywords internal
list_packages <- function(envname = "BFR") {
  reticulate::py_list_packages(envname = envname)
}

#' @title Install dependencies
#' @details
#' Kept for backward compatibility; same as [update_BF()].
#' @param envname A character string specifying the name of the virtual environment.
#' @keywords internal
install_dependencies <- function(envname = "BFR") {
  update_BF(envname = envname)
}

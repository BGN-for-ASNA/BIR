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

#' Check if the default virtual environment is available
#'
#' This function checks for the existence of the default Python virtual environments.
#'
#' @return A logical value indicating whether the "cpu" or "gpu" environment exists.
#' @keywords internal
check_env <- function() {
  cpu_env_exists <- reticulate::virtualenv_exists(envname = "BFR")


  if (cpu_env_exists) {
    packageStartupMessage("Virtual environment ('BFR') is available.")
  } else {
    packageStartupMessage("Virtual environment ('BFR') is not available.")
  }

  return(cpu_env_exists)
}

#' Create a Python virtual environment
#'
#' This function creates a Python virtual environment using reticulate.
#'
#' @param backend A character string specifying the backend, either "cpu" or "gpu".
#'
#' @return The path to the created virtual environment.
#' Create a Python virtual environment and install dependencies
#'
#' This function creates a Python virtual environment and installs a
#' specified set of Python packages, with options for CPU or GPU builds.
#'
#' @param backend A character string specifying the backend, either "cpu" or "gpu".
#' @param env_name A character string for the name of the virtual environment.
#'                 Defaults to "BFR-cpu" or "BFR-gpu".
#'
#' @return The path to the created virtual environment.
#' @export
setup_env <- function(env_name = "BFR", backend = "cpu") {
  if (!backend %in% c("cpu", "gpu")) {
    stop("backend must be either 'cpu' or 'gpu'")
  }
  env_name <- paste0("BFR")

  # Base dependencies
  base_packages <- c("arviz", "numpyro", "IPython")

  # Backend-specific dependencies
  if (backend == "cpu") {
    backend_packages <- c(
      "jax==0.6.2",
      "jaxlib==0.6.2"
    )
  } else { # gpu
    backend_packages <- "jax[cuda12_pip]==0.6.2"
  }

  all_packages <- c(base_packages, backend_packages)
  # all_packages <- c(backend_packages)
  # all_packages <- c(base_packages)

  if (!reticulate::virtualenv_exists(envname = env_name)) {
    packageStartupMessage(paste("Creating virtual environment:", env_name))
    reticulate::virtualenv_create(envname = env_name, packages = all_packages)
  } else {
    packageStartupMessage(paste("Virtual environment", env_name, "already exists."))
    packageStartupMessage(paste("Instaling BI dependencies : jax, arviz, numpyro"))
    reticulate::py_install(
      packages = all_packages,
      envname = env_name,
      pip = TRUE,
      pip_options = "--upgrade" # To ensure latest packages
    )
  }

  return(reticulate::virtualenv_python(envname = env_name))
}

#' @title Run a starting test for the BI environment
#' @details
#' Internal function that checks whether Python and the `bayesforge`
#' package are available, and sets internal `.BF_env` flags accordingly.
#'
#' @details
#' Checks if the `reticulate` package is installed, if Python is available,
#' and if the `bayesforge` package exists in the target environment.
#' Updates `.BF_env$loaded` and `.BF_env$.bf` accordingly.
#'
#' @keywords internal
BF_starting_test <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    packageStartupMessage("The 'reticulate' package is required but not installed")
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    packageStartupMessage("Python is not available on this system. Please install Python before using this package.")
  }

  BI_venv_present <- check_env()


  if (!BI_venv_present) {
    packageStartupMessage("No Python virtual environments found.")
    packageStartupMessage("You can create one manually with : setup_env().")
    ask_yes_no("Do you want to install a clean version of python. (Y/N): ") -> answer
    if (answer == "Y") {
      reticulate::install_python(version = "3.11:latest")
    }
    ask_yes_no("Alternativelly, we can do it now and install required depedencies for Bayesian Inference. (Y/N): ") -> answer
    if (answer == "Y") {
      ask_yes_no("Do you want 'gpu' backend? Note that GPU backend is only available for Linux or WSL2  (Y/N): ") -> answer2
      if (answer2 == "Y") {
        packageStartupMessage("Creating GPU virtual environment...")
        backend <- "gpu"
      } else {
        packageStartupMessage("Creating CPU virtual environment...")
        backend <- "cpu"
      }
      setup_env(backend = backend)
      packageStartupMessage("Virtual environment created, Instaling BI dependencies : jax, arviz, numpyro")
      BI_venv_present <- check_env()
      if (BI_venv_present) {
        reticulate::py_install(
          packages = "bayesforge",
          envname = "BFR",
          pip = TRUE,
          pip_options = "--upgrade" # To ensure latest packages
        )
        packageStartupMessage("Virtual environment setup. You need to restart R session.")
      } else {
        packageStartupMessage("There was an issue creating the virtual environment. Please try again.")
      }
      .BF_env$loaded <- TRUE
      return(invisible(NULL))
    } else {
      packageStartupMessage(paste("Using", reticulate::virtualenv_list()))
      reticulate::use_virtualenv("BFR", required = TRUE)
    }
  } else {
    packageStartupMessage("Using 'BFR' virtual environment.")
    tmp <- reticulate::virtualenv_list()
    if (length(tmp) > 1) {
      packageStartupMessage(paste("Multiple virtual environments found:", paste(tmp, collapse = ", ")))
      packageStartupMessage("Using 'BFR' virtual environment.")
    } else {
      if (reticulate::py_available(initialize = FALSE)) {
        # packageStartupMessage("You need to restart R session!")
      } else {
        reticulate::use_virtualenv("BFR", required = TRUE)
      }
    }

    r <- reticulate::py_list_packages("BFR")
    r <- r[r[, 1] %in% c("numpyro", "arviz", "jax"), ]
    r[, 2] == c("0.22.0", "0.6.2", "0.19.0")
  }
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


#' @title Load the BI module from Python
#' @details
#' Internal helper function to import the `BI` Python module.
#'
#' @return The Python object corresponding to `BayesForge$bf`.
#' @keywords internal
BF_load <- function() {
  tryCatch(
    {
      reticulate::import("BayesForge")$bf
    },
    error = function(e) {
      message("\n----------------------------------------------------")
      message("An error occurred: ", e$message)
      message("----------------------------------------------------")
    }
  )
}

#' @title Update the BI module from Python
#' @details
#' Internal helper function to update the `BI` Python module.
#' @param envname A character string specifying the name of the virtual environment.
#' @keywords internal
update_BF <- function(envname = "BFR") {
  reticulate::py_install(
    packages = "bayesforge",
    envname = envname,
    pip_args = "--upgrade --no-cache-dir"
  )
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
#' Internal helper function to install dependencies for the `BFR` package.
#' @param envname A character string specifying the name of the virtual environment.
#' @keywords internal
install_dependencies <- function(envname = "BFR") {
  reticulate::py_install(
    packages = "numpyro==0.19.0",
    envname = envname,
    pip_args = "--upgrade --no-cache-dir"
  )
  reticulate::py_install(
    packages = "jax==0.6.2",
    envname = envname,
    pip_args = "--upgrade --no-cache-dir"
  )

  reticulate::py_install(
    packages = "bayesforge",
    envname = envname,
    pip_args = "--upgrade --no-cache-dir"
  )
}

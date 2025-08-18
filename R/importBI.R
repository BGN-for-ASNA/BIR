#' @title Import the BI Python Module
#'
#' @description
#' This function initializes the BI Python module through **reticulate**,
#' sets up the environment, and loads the necessary `jax` and `jax.numpy`
#' modules. The BI module is stored in the hidden object `.bi` for internal use,
#' but the initialized BI object is also returned for convenience.
#'
#' @param platform Character string, the computational platform to use
#'   (e.g. `"cpu"` or `"gpu"`). Defaults to `"cpu"`.
#' @param cores Integer or `NULL`. Number of CPU cores to use. Defaults to `NULL`.
#' @param deallocate Logical. Whether memory should be deallocated when not in use.
#'   Defaults to `FALSE`.
#'
#' @return An initialized BI module object (Python object via **reticulate**).
#'
#' @details
#' - Internally, this function imports the `BI` Python package and assigns it
#'   to the hidden variable `.bi`.
#' - It also imports `jax` and `jax.numpy`, assigning them to `jax` and `jnp`
#'   respectively.
#' - Startup messages inform the user about the imports.
#' @examples
#' library(BayesianInference)
#' m = importBI()
#'@export
#'
importBI <- function(platform = 'cpu', cores = NULL, deallocate = FALSE) {
  # Import the main BI module and its 'bi' class
  .BI_env$.bi <- reticulate::import("BI")$bi
  packageStartupMessage("BIR loaded BI into its internal environment.")

  # Import jax and jax.numpy
  .BI_env$jax <- reticulate::import('jax')
  .BI_env$jnp <- reticulate::import('jax.numpy')
  packageStartupMessage("jax and jax.numpy have been imported.")

  # Initialize the BI class
  .BI_env$.bi_instance <- .BI_env$.bi(platform = platform,
                                        cores = reticulate::r_to_py(cores),
                                        deallocate = reticulate::r_to_py(deallocate))

  # A helper Python function if needed
  .BI_env$.py <- reticulate::py_run_string("def is_none(x): return x is None")

  invisible(.BI_env$.bi_instance)
}

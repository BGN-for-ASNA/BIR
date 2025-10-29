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
#' @param  rand_seed (Boolean): Random seed. Defaults to TRUE.
#' @param deallocate Logical. Whether memory should be deallocated when not in use.
#'   Defaults to `FALSE`.
#' @param print_devices_found (bool, optional): Whether to print devices found. Defaults to TRUE.
#' @param backend (str, optional): Backend to use (numpyro or tfp). Defaults to 'numpyro'.
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
#' \dontrun{
#' library(BayesianInference)
#' m = importBI()
#' }
#'@export
#'
importBI <- function(
    platform = 'cpu', 
    cores = NULL, 
    rand_seed = TRUE,
    deallocate = FALSE,
    print_devices_found = TRUE, 
    backend='numpyro'
    ) {
  message("\n----------------------------------------------------")
  message("Loading BI")
  message("----------------------------------------------------")
  tryCatch({
    BI_starting_test()
    .BI_env$.bi <- BI_load()
    .BI_env$loaded <- TRUE
  }, error = function(e){
    message("\n----------------------------------------------------")
    message("An error occurred: ", e$message)
    message("----------------------------------------------------")
  })
  
  # Import jax and jax.numpy
  #.BI_env$jax <- reticulate::import('jax')
  .BI_env$jnp <- reticulate::import('jax.numpy')
  packageStartupMessage("jax and jax.numpy have been imported.")
  
  # Initialize the BI class
  .BI_env$.bi_instance <- .BI_env$.bi(platform = platform,
                                      cores = reticulate::r_to_py(cores),
                                      rand_seed = reticulate::r_to_py(rand_seed),
                                      deallocate = reticulate::r_to_py(deallocate),
                                      print_devices_found = = reticulate::r_to_py(print_devices_found), 
                                      backend = reticulate::r_to_py(backend),
                                      )
  
  # A helper Python function if needed
  .BI_env$.py <- reticulate::py_run_string("def is_none(x): return x is None")
  
  invisible(.BI_env$.bi_instance)
}

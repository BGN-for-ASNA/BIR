#' @title Start a BayesForge session
#'
#' @description
#' Binds R to the BayesForge Python environment (created once with
#' [setup_env()]), imports the `BayesForge` Python module through
#' **reticulate** and returns an initialised BayesForge object.
#'
#' @param platform Character string, the computational platform to use
#'   (e.g. `"cpu"` or `"gpu"`). Defaults to `"cpu"`.
#' @param cores Integer or `NULL`. Number of CPU cores to use. Defaults to `NULL`.
#' @param rand_seed Logical or integer. An integer gives reproducible results;
#'   `TRUE` draws an entropy-based seed. Defaults to `TRUE`.
#' @param deallocate Logical. Whether memory should be deallocated when not in use.
#'   Defaults to `FALSE`.
#' @param print_devices_found Logical. Whether to print devices found. Defaults to `TRUE`.
#' @param backend Character. Backend to use (`"numpyro"` or `"tfp"`). Defaults to `"numpyro"`.
#'
#' @return An initialised BayesForge object (Python object via **reticulate**).
#'
#' @details
#' - If the "BFR" Python environment does not exist yet, interactive sessions
#'   are offered to create it with [setup_env()]; otherwise an error explains
#'   how to do it. An existing environment is never reinstalled.
#' - `jnp` (`jax.numpy`) and `jax` are available as soon as the package is
#'   loaded, e.g. `jnp$array(c(1, 2, 3))`.
#' @examples
#' \dontrun{
#' library(BayesForge)
#' m <- importBF()
#' }
#' @export
#'
importBF <- function(
    platform = "cpu",
    cores = NULL,
    rand_seed = TRUE,
    deallocate = FALSE,
    print_devices_found = TRUE,
    backend = "numpyro") {
  BF_starting_test()
  .BF_env$.bf <- BF_load()
  .BF_env$loaded <- TRUE
  .BF_env$jnp <- reticulate::import("jax.numpy")

  # R numbers are doubles; the Python side expects integers for these.
  if (is.numeric(rand_seed)) rand_seed <- as.integer(rand_seed)
  if (is.numeric(cores)) cores <- as.integer(cores)

  .BF_env$.bf_instance <- .BF_env$.bf(
    platform = platform,
    cores = reticulate::r_to_py(cores),
    rand_seed = reticulate::r_to_py(rand_seed),
    deallocate = reticulate::r_to_py(deallocate),
    print_devices_found = reticulate::r_to_py(print_devices_found)
  )

  # A helper Python function if needed
  .BF_env$.py <- reticulate::py_run_string("def is_none(x): return x is None")
  invisible(.BF_env$.bf_instance)
}

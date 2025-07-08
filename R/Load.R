.onLoad <- function(libname, pkgname) {
  usethis::use_package("cli")
  module_is_available <- reticulate::py_module_available("BI")


  if (!module_is_available) {
    # Use cli to format a nice warning message
    cli::cli_div(theme = list(rule = list(color = "yellow")))

    cli::cli_h1("Python Dependency Missing")

    cli::cli_alert_warning(
      "The required Python package {.pkg BayesInference} is not installed."
    )

    cli::cli_text(
      "Please install it in your R session by running the following command:"
    )


    # This will be formatted as a nice code block
    cli::cli_code("reticulate::py_install('BayesInference')")

    cli::cli_text(
      "For Linux or WSL2 users, you can install the CUDA version of the package by running the following command:"
    )

    # This will be formatted as a nice code block
    cli::cli_code("reticulate::py_install('BayesInference[cuda12]')")

    cli::cli_alert_info(
      "For more details on {.pkg BayesInference}, see {.url https://github.com/BGN-for-ASNA/BIR}"
    )

    cli::cli_rule() # A closing horizontal rule
  }
}

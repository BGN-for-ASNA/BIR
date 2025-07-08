bi.doc <- function() {
  # browseURL(paste(system.file(package = "BI"), "/BI/resources/documentation/0. Introduction.html", sep = ''))

  cli::cli_div(theme = list(rule = list(color = "yellow")))

  cli::cli_h1("Documentation Implementation")

  cli::cli_alert_warning(
    "This is a pre-release. The documentation is not yet available within the package, as it is currently in development."
  )

  cli::cli_alert_warning(
    "To access the development version of the documentation, visit {.url https://github.com/BGN-for-ASNA/BI}"
  )
}

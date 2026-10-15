#' @keywords internal
.bf_is_plotly <- function(obj) {
  if (!reticulate::is_py_object(obj)) return(FALSE)
  tryCatch({
    mod <- reticulate::py_get_attr(
      reticulate::py_get_attr(obj, "__class__"), "__module__"
    )
    grepl("^plotly", as.character(mod))
  }, error = function(e) FALSE)
}

#' @keywords internal
.bf_show_plotly <- function(fig) {
  tmp <- tempfile(fileext = ".html")
  fig$write_html(tmp)
  utils::browseURL(tmp)
  invisible(fig)
}

#' Wrap a BayesForge plot namespace so every method auto-displays plotly figures.
#'
#' @param py_ns A Python object (e.g. \code{m$diag}, \code{m$ppc}).
#' @return An environment of class \code{"bf_plot_ns"} where calling any method
#'   automatically saves the returned plotly Figure to a temp HTML file and
#'   opens it in the browser.
#' @export
bf_plot_ns <- function(py_ns) {
  methods <- tryCatch(
    reticulate::py_list_attributes(py_ns),
    error = function(e) character(0)
  )
  methods <- methods[!startsWith(methods, "_")]

  e <- new.env(parent = emptyenv())
  for (nm in methods) {
    local({
      name <- nm
      fn   <- tryCatch(py_ns[[name]], error = function(e) NULL)
      if (is.null(fn) || !reticulate::is_py_object(fn)) return()
      is_fn <- tryCatch(
        reticulate::py_has_attr(fn, "__call__"),
        error = function(e) FALSE
      )
      if (!is_fn) return()
      force(fn)
      e[[name]] <- function(...) {
        result <- fn(...)
        if (.bf_is_plotly(result)) .bf_show_plotly(result)
        else result
      }
    })
  }
  structure(e, class = "bf_plot_ns")
}

#' Display a BayesForge plotly figure in the browser.
#'
#' Saves the figure to a temporary HTML file and opens it.
#'
#' @param fig A plotly Figure returned by a BayesForge diagnostic function.
#' @return The figure, invisibly.
#' @export
bf_show <- function(fig) {
  if (!.bf_is_plotly(fig)) {
    warning("bf_show: object does not appear to be a plotly Figure.")
    return(invisible(fig))
  }
  .bf_show_plotly(fig)
}

#' Print a BayesForge plot namespace
#'
#' @param x A `bf_plot_ns` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.bf_plot_ns <- function(x, ...) {
  fns <- ls(envir = x)
  cat("<BayesForge plot namespace>\n")
  cat("Available methods:", paste(fns, collapse = ", "), "\n")
  invisible(x)
}

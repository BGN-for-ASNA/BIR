#' @title title Open the BF Documentation Website
#' @description
#' This function opens the BF documentation webpage in the user's default
#' web browser.
#'
#' @return Invisibly returns `NULL`. The function is called for its side effect
#' of opening a URL.
#' @export
bf.doc <- function() {
  browseURL("https://s-sosa.com/BF/")
}

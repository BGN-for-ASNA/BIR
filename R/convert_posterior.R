#' @title Convert Posterior Samples
#' @description
#' Converts posterior samples from Python to R format.
#' @param posteriors Posterior samples from Python model.
#' @return A list of posterior samples in R format.
#' @keywords internal
convert_posterior <- function(posteriors) {
  np <- reticulate::import("numpy")
  R_list <- reticulate::py_to_r(posteriors)
  for (a in 1:length(R_list)) {
    R_list[[a]] <- reticulate::py_to_r(np$array(R_list[[a]]))
  }
  return(R_list)
}


convert_posterior = function(posteriors){
  np = reticulate::import('numpy')
  R_list = reticulate::py_to_r(posteriors)
  for(a in 1:length(R_list)){
    R_list[[a]] = reticulate::py_to_r(np$array(R_list[[a]]))
  }
  return(R_list)
}

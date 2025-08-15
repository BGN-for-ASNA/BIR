importBI = function(platform='cpu', cores=NULL, deallocate = FALSE){
  # Construct the path to the directory containing main.py
  .bi <<- import("BI")
  .bi <<-  .bi$bi
  packageStartupMessage("BIR load BI as .bi do not overwrite it!")

  # Import the main module from the
  jax<<-import('jax')
  jnp<<-import('jax.numpy')
  packageStartupMessage("jax an jax.numpy have been imported as jax and jnp respectivelly")
  .bi <<- .bi(platform=platform, cores=reticulate::r_to_py(cores),
              deallocate = reticulate::r_to_py(deallocate))
  m=.bi
  return(m)
}


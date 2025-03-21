
#importBI = function(platform='cpu', cores=NULL, deallocate = FALSE){
#  require(reticulate)
#  # Construct the path to the directory containing main.py
#  module_dir <- system.file("BI", package = "BI")

#  # Import the main module from the specified directory

#  .bi <<- reticulate::import_from_path("main", path = module_dir)
#  jax<<-import('jax')
#  jnp<<-import('jax.numpy')

#  packageStartupMessage("BIR load BI as .bi do not overwrite it!")
#  packageStartupMessage("jax an jax.numpy have been imported as jax and jnp respectivelly")
#  m = .bi$bi(platform=platform, cores=reticulate::r_to_py(cores),
#             deallocate = reticulate::r_to_py(deallocate))
#  return(m)
#}


importBI = function(platform='cpu', cores=NULL, deallocate = FALSE){
  require(reticulate)

  # Construct the path to the directory containing main.py
  bi_path <- system.file(package = "BI")
  .bi <<- reticulate::import_from_path("BI", path = bi_path)
  .bi <<-  .bi$Main$main$bi

  # Import the main module from the
  jax<<-import('jax')
  jnp<<-import('jax.numpy')

  packageStartupMessage("BIR load BI as .bi do not overwrite it!")
  packageStartupMessage("jax an jax.numpy have been imported as jax and jnp respectivelly")
  m = .bi(platform=platform, cores=reticulate::r_to_py(cores),
             deallocate = reticulate::r_to_py(deallocate))

  .NBDA <- import("BI.NBDA")
  m$nbda = .NBDA$nbda$NBDA
  return(m)
}






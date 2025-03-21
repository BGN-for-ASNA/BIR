to.jnp.shape <- function(shape){
  # COnvert R shape to compatiple python jno shape
  do.call(tuple, as.list(as.integer(shape)))
}

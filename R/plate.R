
#' @export
bi.dist.plate=function(name, shape) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$plate(name,  shape)
}

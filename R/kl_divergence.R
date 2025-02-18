bi.dist.kl_divergence=function(shape=c(), sample=FALSE, seed=0, name='x', obs=py_none(), ...) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$kl_divergence(shape=c(), sample=FALSE, seed=0, name='x', obs=py_none(), list(...))
}

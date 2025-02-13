bi.dist.kl_divergence=function(shape=c(), sample=FALSE, seed=0, name='x', ...) { 
     bi=importBI(platform='cpu');
    shape=do.call(tuple, as.list(as.integer(shape)));
    seed=as.integer(seed);
bi$dist$kl_divergence(shape=c(), sample=FALSE, seed=0, name='x', list(...))}

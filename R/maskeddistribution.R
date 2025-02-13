bi.dist.maskeddistribution=function(base_dist, mask, shape=c(), sample=FALSE, seed=0, name='x') { 
     bi=importBI(platform='cpu');
    shape=do.call(tuple, as.list(as.integer(shape)));
    seed=as.integer(seed);
bi$dist$maskeddistribution(base_dist,  mask,  shape= shape,  sample= sample,  seed= seed,  name= name)}

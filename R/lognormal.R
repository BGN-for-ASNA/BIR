bi.dist.lognormal=function(loc=0.0, scale=1.0, validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x') { 
     bi=importBI(platform='cpu');
    shape=do.call(tuple, as.list(as.integer(shape)));
    seed=as.integer(seed);
bi$dist$lognormal(loc=loc,  scale= scale,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name)}

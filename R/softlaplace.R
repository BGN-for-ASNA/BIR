bi.dist.softlaplace=function(loc, scale, validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x', obs=py_none()) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$softlaplace(loc,  scale,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name,  obs= obs)
}

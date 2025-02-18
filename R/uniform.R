bi.dist.uniform=function(low=0.0, high=1.0, validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x', obs=py_none()) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$uniform(low=low,  high= high,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name,  obs= obs)
}

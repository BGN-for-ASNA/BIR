bi.dist.binomiallogits=function(logits, total_count=1, validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x') { 
     bi=importBI(platform='cpu');
    shape=do.call(tuple, as.list(as.integer(shape)));
    seed=as.integer(seed);
bi$dist$binomiallogits(logits,  total_count= total_count,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name)}

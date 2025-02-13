bi.dist.mixturesamefamily=function(mixing_distribution, component_distribution, validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x') { 
     bi=importBI(platform='cpu');
    shape=do.call(tuple, as.list(as.integer(shape)));
    seed=as.integer(seed);
bi$dist$mixturesamefamily(mixing_distribution,  component_distribution,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name)}

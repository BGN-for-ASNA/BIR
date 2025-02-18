bi.dist.gaussianstatespace=function(num_steps, transition_matrix, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), shape=c(), sample=FALSE, seed=0, name='x', obs=py_none()) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$gaussianstatespace(num_steps,  transition_matrix,  covariance_matrix= covariance_matrix,  precision_matrix= precision_matrix,  scale_tril= scale_tril,  validate_args= validate_args,  shape= shape,  sample= sample,  seed= seed,  name= name,  obs= obs)
}

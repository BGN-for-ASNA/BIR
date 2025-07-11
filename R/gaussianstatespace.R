#' @title GaussianStateSpace distribution wrapper.
#' @param num_steps <class 'inspect._empty'>
#' @param transition_matrix <class 'inspect._empty'>
#' @param covariance_matrix None
#' @param precision_matrix None
#' @param scale_tril None
#' @param validate_args None
#' @param shape (tuple): A multi-purpose argument for shaping. - When sample=False (model building), this is used with `.expand(shape)` to set the distribution's batch shape. - When sample=True (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (jnp.ndarray, bool): Optional boolean array to mask observations. This is passed to the `infer={'obs_mask': ...}` argument of `numpyro.sample`.
#' @param create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @examples
#' bi.dist.gaussianstatespace(sample = TRUE)
#' @export
bi.dist.gaussianstatespace=function(num_steps, transition_matrix, covariance_matrix=py_none(), precision_matrix=py_none(), scale_tril=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$gaussianstatespace(num_steps,  transition_matrix,  covariance_matrix= covariance_matrix,  precision_matrix= precision_matrix,  scale_tril= scale_tril,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

#' @title GaussianCopulaBeta distribution wrapper.
#' @param concentration1 <class 'inspect._empty'>
#' @param concentration0 <class 'inspect._empty'>
#' @param correlation_matrix None
#' @param correlation_cholesky None
#' @param validate_args False
#' @param shape (tuple): A multi-purpose argument for shaping. - When sample=False (model building), this is used with `.expand(shape)` to set the distribution's batch shape. - When sample=True (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (jnp.ndarray, bool): Optional boolean array to mask observations. This is passed to the `infer={'obs_mask': ...}` argument of `numpyro.sample`.
#' @param create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @examples
#' bi.dist.gaussiancopulabeta(sample = TRUE)
#' @export
bi.dist.gaussian_copula_beta=function(concentration1, concentration0, correlation_matrix=py_none(), correlation_cholesky=py_none(), validate_args=FALSE, name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     if(py$is_none(correlation_cholesky)){
        .bi$dist$gaussian_copula_beta(
          concentration1 = jnp$array(concentration1),
          concentration0 = jnp$array(concentration0),
          correlation_matrix= jnp$array(correlation_matrix),
          validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
     }else{
       .bi$dist$gaussian_copula_beta(
         concentration1 = jnp$array(concentration1),
         concentration0 = jnp$array(concentration0),
         correlation_cholesky= jnp$array(correlation_cholesky),
         validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
     }
}

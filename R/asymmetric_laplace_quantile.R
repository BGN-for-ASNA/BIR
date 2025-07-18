#' @title AsymmetricLaplaceQuantile distribution wrapper.
#' @param loc 0.0
#' @param scale 1.0
#' @param quantile 0.5
#' @param validate_args None
#' @param shape (tuple): A multi-purpose argument for shaping. - When sample=False (model building), this is used with `.expand(shape)` to set the distribution's batch shape. - When sample=True (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (jnp.ndarray, bool): Optional boolean array to mask observations. This is passed to the `infer={'obs_mask': ...}` argument of `numpyro.sample`.
#' @param create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @examples
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.asymmetriclaplacequantile(sample = TRUE)
#' @export
bi.dist.asymmetric_laplace_quantile=function(loc=0.0, scale=1.0, quantile=0.5, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$asymmetric_laplace_quantile(loc=loc,  scale= scale,  quantile= quantile,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

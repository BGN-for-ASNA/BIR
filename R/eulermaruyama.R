#' @title EulerMaruyama distribution wrapper.
#' @param t <class 'inspect._empty'>
#' @param sde_fn <class 'inspect._empty'>
#' @param init_dist <class 'inspect._empty'>
#' @param validate_args None
#' @param shape (tuple): A multi-purpose argument for shaping. - When sample=False (model building), this is used with `.expand(shape)` to set the distribution's batch shape. - When sample=True (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
#' @param event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask (jnp.ndarray, bool): Optional boolean array to mask observations. This is passed to the `infer={'obs_mask': ...}` argument of `numpyro.sample`.
#' @param create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
#' @examples
#' library(BI)
#' m=importBI(platform='cpu')
#' bi.dist.eulermaruyama(0, 1, .bi$dist$normal(0,1,create_obj = T), sample = TRUE, event = c(1,10))
#' @export
bi.dist.eulermaruyama=function(t, sde_fn, init_dist, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) {
     shape=do.call(tuple, as.list(as.integer(shape)))
     event=as.integer(event)
     seed=as.integer(seed);
     .bi$dist$eulermaruyama(t,  sde_fn,  init_dist,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}

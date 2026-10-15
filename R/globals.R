# Empty file to avoid duplicate definition and overriding .BF_env from Load.R

# `enum = TRUE` on a discrete distribution = NumPyro's
# infer = list(enumerate = "parallel"); an explicit `infer` wins key by key.
# Translated in R so that it works with every BayesForge (Python) version.
.bf_infer <- function(infer, enum) {
  if (!isTRUE(enum)) return(infer)
  base <- list(enumerate = "parallel")
  if (is.null(infer) || inherits(infer, "python.builtin.NoneType")) return(base)
  utils::modifyList(base, as.list(infer))
}

# Name of an observed site: the R variable passed as `obs` (obs = y -> "y");
# "obs" for an expression (obs = y - 1). Used when `name` is not given, so an
# observed site no longer defaults to "x" and collides with a parameter x.
.bf_obs_name <- function(expr) {
  if (is.name(expr)) deparse(expr) else "obs"
}

# jnp$asarray() for an argument that may be left at py_none() (e.g. the
# bounds of a truncated distribution): None is passed through unchanged.
.bf_arr <- function(x) {
  if (is.null(x) || inherits(x, "python.builtin.NoneType")) reticulate::py_none()
  else .BF_env$jnp$asarray(x)
}

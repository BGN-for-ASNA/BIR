# R math on JAX arrays, so model code needs no jnp$...
#
# reticulate already maps the arithmetic operators (+, -, *, /, ^, %*%,
# comparisons) of Python objects to Python. The S3 group generics below add
# the R math functions: exp(x), log(x), sqrt(x), sum(x), mean(x), max(x)...
# on a JAX array -- or on a JAX tracer, which is what a model sees while it
# is fitted -- now call the jax.numpy function instead of failing with
# "non-numeric argument to mathematical function".

.bf_jnp <- function() reticulate::import("jax.numpy", convert = FALSE)

# R name -> function(x, ...) on a JAX value
.bf_math_fun <- function(generic, x, ...) {
  jnp <- .bf_jnp()
  special <- function(name) reticulate::import("jax.scipy.special", convert = FALSE)[[name]]
  lax <- function() reticulate::import("jax.lax", convert = FALSE)
  args <- list(...)
  switch(generic,
    log      = if (length(args) && !is.null(args$base) && !identical(args$base, exp(1)))
                 jnp$log(x) / jnp$log(args$base) else jnp$log(x),
    ceiling  = jnp$ceil(x),
    round    = jnp$round(x, decimals = as.integer(if (length(args)) args[[1]] else 0L)),
    gamma    = special("gamma")(x),
    lgamma   = special("gammaln")(x),
    digamma  = special("digamma")(x),
    trigamma = special("polygamma")(1L, x),
    cummax   = lax()$cummax(x),
    cummin   = lax()$cummin(x),
    signif   = stop("signif() is not available on JAX arrays."),
    {
      f <- tryCatch(jnp[[generic]], error = function(e) NULL)
      if (is.null(f)) stop(generic, "() is not available on JAX arrays.")
      f(x)
    }
  )
}

.bf_Math <- function(x, ...) .bf_math_fun(.Generic, x, ...)

.bf_Summary <- function(..., na.rm = FALSE) {
  vals <- list(...)
  if (length(vals) != 1L)
    stop(.Generic, "() on JAX arrays takes a single array; combine them with jnp$concatenate().")
  x <- vals[[1]]
  jnp <- .bf_jnp()
  switch(.Generic,
    sum   = jnp$sum(x),
    prod  = jnp$prod(x),
    max   = jnp$max(x),
    min   = jnp$min(x),
    all   = jnp$all(x),
    any   = jnp$any(x),
    range = jnp$stack(list(jnp$min(x), jnp$max(x))),
    stop(.Generic, "() is not available on JAX arrays.")
  )
}

.bf_mean <- function(x, ...) .bf_jnp()$mean(x)

# Registered in NAMESPACE as S3method(Math, jax.Array, .bf_Math), etc.

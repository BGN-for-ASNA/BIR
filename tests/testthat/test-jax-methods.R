if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)
  np <- reticulate::import("numpy", convert = FALSE)
  jnpf <- reticulate::import("jax.numpy", convert = FALSE)
  sp <- reticulate::import("jax.scipy.special", convert = FALSE)
  lax <- reticulate::import("jax.lax", convert = FALSE)
  as_r <- function(v) reticulate::py_to_r(np$asarray(v))

  x <- jnpf$asarray(c(0.3, 1.7, 2.5, 4.2))

  test_that("R math functions on a JAX array equal the jax.numpy result", {
    cases <- list(
      list(exp(x), jnpf$exp(x)), list(log(x), jnpf$log(x)),
      list(log(x, base = 2), jnpf$log(x) / jnpf$log(2)), list(sqrt(x), jnpf$sqrt(x)),
      list(abs(x), jnpf$abs(x)), list(cos(x), jnpf$cos(x)), list(tanh(x), jnpf$tanh(x)),
      list(floor(x), jnpf$floor(x)), list(ceiling(x), jnpf$ceil(x)),
      list(round(x, 1), jnpf$round(x, decimals = 1L)), list(log1p(x), jnpf$log1p(x)),
      list(expm1(x), jnpf$expm1(x)), list(lgamma(x), sp$gammaln(x)),
      list(gamma(x), sp$gamma(x)), list(digamma(x), sp$digamma(x)),
      list(cumsum(x), jnpf$cumsum(x)), list(cumprod(x), jnpf$cumprod(x)),
      list(cummax(x), lax$cummax(x)),
      list(sum(x), jnpf$sum(x)), list(prod(x), jnpf$prod(x)), list(max(x), jnpf$max(x)),
      list(min(x), jnpf$min(x)), list(mean(x), jnpf$mean(x)),
      list(range(x), jnpf$stack(list(jnpf$min(x), jnpf$max(x)))))
    for (cs in cases) expect_identical(as_r(cs[[1]]), as_r(cs[[2]]))
    expect_error(sum(x, x), "single array")
  })

  fit <- function(model) {
    m <- importBF(platform = "cpu", rand_seed = 9, print_devices_found = FALSE)
    t <- seq(0.1, 2, length.out = 25)
    m$fit(model, obs = list(t = jnpf$asarray(t), y = jnpf$asarray(round(3 * exp(0.4 * t)))),
          num_warmup = 100L, num_samples = 100L, num_chains = 1L, progress_bar = FALSE)
    m
  }
  draws <- function(m, k) as_r(m$posteriors[[k]])

  test_that("a model written without jnp$ gives the jnp$ posterior, bit for bit", {
    plain <- fit(function(t, y) {
      a <- bf.dist.normal(0, 1, name = "a")
      b <- bf.dist.normal(0, 1, name = "b")
      rate <- exp(a + b * t) / mean(exp(a + b * t)) * mean(y)   # R functions
      bf.dist.poisson(rate = rate, obs = y)
    })
    with_jnp <- fit(function(t, y) {
      a <- bf.dist.normal(0, 1, name = "a")
      b <- bf.dist.normal(0, 1, name = "b")
      rate <- jnp$exp(a + b * t) / jnp$mean(jnp$exp(a + b * t)) * jnp$mean(y)
      bf.dist.poisson(rate = rate, obs = y)
    })
    for (k in c("a", "b")) expect_identical(draws(plain, k), draws(with_jnp, k))
  })
}

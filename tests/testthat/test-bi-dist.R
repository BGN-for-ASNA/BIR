library(BI)
m=importBI(platform='cpu')

test_that("bi.dist.asymmetric_laplace", {

  res = bi.dist.asymmetric_laplace(sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(round(r2, digits = 4), -0.2983)
})

test_that("bi.dist.asymmetric_laplace_quantile", {

  res = bi.dist.asymmetric_laplace_quantile(sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(round(r2, digits = 4), -0.5967)
})

test_that("bi.dist.bernoulli", {

  res = bi.dist.bernoulli(probs = 0.5, sample = TRUE, seed = 5)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(round(r2, digits = 4), 0.00)
})

test_that("bi.dist.beta", {

  res = bi.dist.beta(concentration1 = 0, concentration0 = 1, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 2.22507386e-308)
})

test_that("bi.dist.beta_binomial", {

  res = bi.dist.beta_binomial(0,1,sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(round(r2, digits = 4), 0.00)
})

test_that("bi.dist.beta_proportion", {

  res =  bi.dist.beta_proportion(0, 1, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 2.225074e-308)
})

test_that("bi.dist.binomial", {

  res =  bi.dist.binomial(probs = jnp$array(c(0.5,0.5)), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2,c(0,0))
})

test_that("bi.dist.car", {

  res =  bi.dist.car(loc = 0,correlation = 0.5, conditional_precision = 0.1, adj_matrix = 2, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2,c(-0.92055403, -3.50957925))
})

test_that("bi.dist.categorical", {

  res = bi.dist.categorical(probs = c(0.5,0.5), sample = TRUE, shape = c(3))
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(0,0,1))
})

test_that("bi.dist.cauchy", {

  res = bi.dist.cauchy(sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, -0.261929506653606)
})

test_that("bi.dist.chi2", {

  res = bi.dist.chi2(0,sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 0.00)
})

test_that("bi.dist.dirichlet", {
  res = bi.dist.dirichlet(concentration = c(0.1,.9),sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(9.98541738e-05, 9.99900146e-01))
})

test_that("bi.dist.dirichlet", {
  res = bi.dist.dirichlet_multinomial(concentration = c(0,1), sample = TRUE, shape = (3))

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, list(c(0,1), c(0,1), c(0,1)))
})

test_that("bi.dist.discrete_uniform", {
  res = bi.dist.discrete_uniform(sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 1)
})

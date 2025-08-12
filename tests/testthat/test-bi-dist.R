library(testthat)
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

test_that("bi.dist.exponential", {
  res = bi.dist.exponential(rate = c(0.1,1,2),sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(5.42070555, 0.24372319, 1.68081713))
})

test_that("bi.dist.gamma_poisson", {
  res = bi.dist.gamma_poisson(concentration = 1, sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 0)
})

test_that("bi.dist.gamma", {
  res =  bi.dist.gamma(concentration = 1 , sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 0.47552933)
})

test_that("bi.dist.gaussian_copula", {
  res =  bi.dist.gaussian_copula(
    marginal_dist = bi.dist.gamma(concentration = 1 ,  create_obj = TRUE) ,
    correlation_matrix =  matrix(c(1.0, 0.7, 0.7, 1.0),, nrow = 2, byrow = TRUE),
    sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(0.542070555, 0.275164930))
})

test_that("bi.dist.gaussian_copula_beta", {
  res =  bi.dist.gaussian_copula_beta(
    concentration1 = c(2.0, 3.0),
    concentration0 = c(5.0, 3.0),
    correlation_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
    sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  r2 = round(r2, digits = 2)
  expect_equal( r2, c(0.23, 0.35))
})

test_that("bi.dist.gaussian_random_walk", {
  res =  bi.dist.gaussian_random_walk(scale = 1 , sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2,-0.205842139479643)
})

test_that("bi.dist.gaussian_random_walk", {
  res =  bi.dist.gaussian_random_walk(scale = 1 , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2,-0.205842139479643)
})

test_that("bi.dist.gaussian_state_space", {
  res =  bi.dist.gaussian_state_space(
    num_steps = 1,
    transition_matrix = matrix(c(0.5), nrow = 1, byrow = TRUE),
    covariance_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
    sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())[[1]]
  expect_equal( r2,c(-0.205842139479643, -0.704524360202176))
})

test_that("bi.dist.geometric", {
  res =  bi.dist.geometric(probs = 0.5 , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2,0.00)
})


test_that("bi.dist.gompertz", {
  res =  bi.dist.gompertz(concentration = 0.5 , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  r2 = round(r2, digits = 4)
  expect_equal(r2, 0.7344)
})

test_that("bi.dist.gumbel", {
  res =  bi.dist.gumbel(loc = 0.5 , scale = 1., sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  r2 = round(r2, digits = 5)
  expect_equal(r2, 0.6379100)
})

test_that("bi.dist.gumbel", {
  res =  bi.dist.gumbel(loc = 0.5 , scale = 1., sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  r2 = round(r2, digits = 5)
  expect_equal(r2, 0.6379100)
})

test_that("bi.dist.half_cauchy", {
  res =  bi.dist.half_cauchy(scale = c(0.5,0.5) , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.13096475, 0.61892301))
})

test_that("bi.dist.half_normal", {
  res =  bi.dist.half_normal(scale = c(0.5,0.5) , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.10292107, 0.39238289))
})

test_that("bi.dist.inverse_gamma", {
  res =  bi.dist.inverse_gamma(concentration = c(0.5,0.5) , sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  r2 = round(r2, digits = 5)
  expect_equal(r2, c(10.19849, 131.85292))
})

test_that("bi.dist.kumaraswamy", {
  res =  bi.dist.kumaraswamy(concentration1 = 1, concentration0 = 10, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 0.083431146)
})

test_that("bi.dist.laplace", {
  res =  bi.dist.laplace(loc = 1, scale = 10, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 2.780336947)
})

test_that("bi.dist.left_truncated_distribution", {
  res =  bi.dist.left_truncated_distribution( base_dist = bi.dist.normal(loc = 1, scale = 10 ,  create_obj = TRUE),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 5.84732542)
})

test_that("bi.dist.levy", {
  res =  bi.dist.levy( loc = 1, scale = 10,  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 16.27547182)
})


test_that("bi.dist.lkj", {
  res =  bi.dist.lkj( dimension = 2, concentration=1.0, shape = c(1), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  lst <- list(
    list(
      c(1.000000000, -0.502239437),
      c(-0.502239437, 1.000000000)
    )
  )
  expect_equal(r2, lst)
})

test_that("bi.dist.lkj_cholesky", {
  res =  bi.dist.lkj_cholesky( dimension = 2, concentration = 1.,  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  r2[[2]] = round(r2[[2]], digits = 5)
  lst <- list(
      c(1, 0.00),
      c(-0.50224000, 0.86473000)
  )

  expect_equal(r2, lst)
})


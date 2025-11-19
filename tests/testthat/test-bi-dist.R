test1 = requireNamespace("reticulate", quietly = TRUE)
test2 = reticulate::py_available(initialize = TRUE)
test3 = reticulate::py_module_available("BI")

if(test1 & test2 & test3){
library(testthat)
library(BayesianInference)
m=importBI(platform='cpu', rand_seed = FALSE)

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
  res = bi.dist.binomial(probs = c(0.5,0.5), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2,c(0,0))
})

test_that("bi.dist.car", {
  res = bi.dist.car(loc = c(1.,2.), correlation = 0.9, conditional_precision = 1., adj_matrix = matrix(c(1,0,0,1), nrow = 2), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2,c(0.34907   , -0.48164728))
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

test_that("bi.dist.delta", {
  res = bi.dist.delta(v = 5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 5.)
})

test_that("bi.dist.dirichlet", {
  res = bi.dist.dirichlet(concentration = c(0.1,.9),sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(9.98541738e-05, 9.99900146e-01))
})

test_that("bi.dist.dirichlet", {
  res = bi.dist.dirichlet_multinomial(concentration = c(0,1), sample = TRUE, shape = c(3))
  r2 = reticulate::py_to_r(res$tolist())

  expect_equal( r2, list(c(0,1), c(0,1), c(0,1)))
})

test_that("bi.dist.discrete_uniform", {
  res = bi.dist.discrete_uniform(sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())

  expect_equal( r2, 1)
})

test_that("bi.dist.euler_maruyama", {
  ornstein_uhlenbeck_sde <- function(x, t) {
    # This function models dX = -theta * X dt + sigma dW
    theta <- 1.0
    sigma <- 0.5

    drift <- -theta * x
    diffusion <- sigma

    # Return a list of two elements: drift and diffusion
    # reticulate will convert this to a Python tuple
    return(list(drift, diffusion))
  }
  res = bi.dist.euler_maruyama(t=c(0.0, 0.1, 0.2),
                               sde_fn = ornstein_uhlenbeck_sde,
                               init_dist=bi.dist.normal(0.0, 1.0, create_obj=TRUE), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, c(-1.4008841 , -0.96353687, -0.94326995))
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
  res =  bi.dist.gamma(concentration = 1 , sample = TRUE, seed = 0)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal( r2, 0.47552933)
})

#test_that("bi.dist.gaussian_copula", {
#  res =  bi.dist.gaussian_copula(
#    marginal_dist = bi.dist.gamma(concentration = 1 ,  create_obj = TRUE) ,
#    correlation_matrix =  matrix(c(1.0, 0.7, 0.7, 1.0),, nrow = 2, byrow = TRUE),
#    sample = TRUE)
#
#  r2 = reticulate::py_to_r(res$tolist())
#  expect_equal( r2, c(0.0840605, 0.7273357))
#})

#test_that("bi.dist.gaussian_copula_beta", {
#  res =  bi.dist.gaussian_copula_beta(
#    concentration1 = c(2.0, 3.0),
#    concentration0 = c(5.0, 3.0),
#    correlation_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
#    sample = TRUE)
#
#  r2 = reticulate::py_to_r(res$tolist())
#  r2 = round(r2, digits = 2)
#  expect_equal( r2, c(0.08, 0.51))
#})


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
    covariance_matrix = matrix(c(1.0), nrow = 1, byrow = TRUE),
    sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())[[1]]
  expect_equal( r2,c(-0.20584214))
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
  res =  bi.dist.laplace(loc = 1, scale = 10, sample = TRUE, seed = 0)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 2.78033695)
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

test_that("bi.dist.log_uniform", {
  res =  bi.dist.log_uniform( low = c(1,1), high = c(10,10),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(1,1))
})


test_that("bi.dist.logistic", {
  res =  bi.dist.logistic( loc = c(1,1), scale = c(10,10),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(-2.2911032 , -11.87386776))
})


test_that("bi.dist.log_normal", {
  res =  bi.dist.log_normal( loc = c(1,1), scale = c(10,10),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.34700316, 0.00106194))
})

test_that("bi.dist.log_normal", {
  res =  bi.dist.log_normal( loc = c(1,1), scale = c(10,10),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.34700316, 0.00106194))
})

test_that("bi.dist.log_normal", {
  res =  bi.dist.log_normal( loc = c(1,1), scale = c(10,10),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.34700316, 0.00106194))
})
jnp <- reticulate::import("jax.numpy")
test_that("bi.dist.low_rank_multivariate_normal", {
  event_size = 10
  rank = 5

  res =  bi.dist.low_rank_multivariate_normal(
    loc = bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)*2,
    cov_factor = bi.dist.normal(0,1,shape = c(event_size, rank), sample = TRUE),
    cov_diag = jnp$exp(bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)),
    sample = TRUE)

  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(-0.52951568, -1.03015152, 13.91182143, -0.85983465,  1.46486219,
                     -3.64957438, -3.16951829, -0.7729603 , -5.94243277, -0.94424 ))
})

test_that("bi.dist.lower_truncated_power_law", {
  res =  bi.dist.lower_truncated_power_law( alpha = c(-2, 2), low = c(1, 0.5),  sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(1.71956363, 0.46098571))
})

test_that("bi.dist.matrix_normal", {
  n_rows= 3
  n_cols = 4
  loc = matrix(rep(0,n_rows*n_cols), nrow = n_rows, ncol = n_cols,byrow = TRUE)

  U_row_cov = jnp$array(matrix(c(1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0), nrow = n_rows, ncol = n_rows,byrow = TRUE))
  scale_tril_row = jnp$linalg$cholesky(U_row_cov)

  V_col_cov = jnp$array(matrix(c(2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1, 0.2, 2.0, 0.0, 0.4, -0.2, 0.0, 2.0), nrow = n_cols, ncol = n_cols,byrow = TRUE))
  scale_tril_column = jnp$linalg$cholesky(V_col_cov)


  res =  bi.dist.matrix_normal( loc = loc, scale_tril_row = scale_tril_row, scale_tril_column = scale_tril_column, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  lst <- list(
      c(-0.291104745, -0.900730803, 2.383118964, 0.207682007),
      c(-0.04650985, -0.90767574,  2.58010087,  0.52933797),
      c(0.08241693, -1.42376624,  2.10489774, -1.86380361)
  )
  expect_equal(r2, lst)
})

test_that("bi.dist.mixture", {
  res = bi.dist.mixture(
    mixing_distribution = bi.dist.categorical(probs = c(0.3, 0, 7),create_obj = TRUE),
    component_distributions = c(bi.dist.normal(0,1,create_obj = TRUE), bi.dist.normal(0,1,create_obj = TRUE), bi.dist.normal(0,1,create_obj = TRUE)),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, -0.24240651)
})

test_that("bi.dist.mixture_general", {
  res = bi.dist.mixture_general(
    mixing_distribution = bi.dist.categorical(probs = c(0.3, 0, 7),create_obj = TRUE),
    component_distributions = c(bi.dist.normal(0,1,create_obj = TRUE), bi.dist.normal(0,1,create_obj = TRUE), bi.dist.normal(0,1,create_obj = TRUE)),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, -0.24240651)
})


test_that("bi.dist.mixture_same_family", {
  res = bi.dist.mixture_same_family(
    mixing_distribution = bi.dist.categorical(probs = c(0.3, 0.7),create_obj = TRUE),
    component_distribution = bi.dist.normal(0,1, shape = c(2), create_obj = TRUE),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 1.88002989)
})

test_that("bi.dist.multinomial_logits", {
  res = bi.dist.multinomial_logits(
    logits =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2, 5, 3))
})


test_that("bi.dist.multinomial_logits", {
  res = bi.dist.multinomial_logits(
    logits =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2, 5, 3))
})

test_that("bi.dist.multinomial_probs", {
  res = bi.dist.multinomial_probs(
    probs =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(1,3,6))
})

test_that("bi.dist.multivariate_normal", {
  res = bi.dist.multivariate_normal(
    loc =  c(1.0, 0.0, -2.0),
    covariance_matrix = matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5), nrow = 3, byrow = TRUE),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.708895254639994, -0.783775419788299, -0.713927354638493))
})

test_that("bi.dist.multivariate_student_t", {
  res = bi.dist.multivariate_student_t(
    df = 2,
    loc =  c(1.0, 0.0, -2.0),
    scale_tril = jnp$linalg$cholesky(matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5), nrow = 3, byrow = TRUE)),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2.91015292,  0.36815279, -2.23324296))
})

test_that("bi.dist.multivariate_student_t", {
  res = bi.dist.multivariate_student_t(
    df = 2,
    loc =  c(1.0, 0.0, -2.0),
    scale_tril = jnp$linalg$cholesky(matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5), nrow = 3, byrow = TRUE)),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2.91015292,  0.36815279, -2.23324296))
})


test_that("bi.dist.multinomial", {
  res = bi.dist.multinomial(
    logits =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2, 5, 3))
})


test_that("bi.dist.negative_binomial_logits", {
  res = bi.dist.negative_binomial_logits(
    logits =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(8.0, 17.0, 22.0))
})

test_that("bi.dist.negative_binomial_probs", {
  res = bi.dist.negative_binomial_probs(
    probs =  c(0.2, 0.3, 0.5),
    total_count = 10,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2.0,  6.0, 14.))
})

test_that("bi.dist.negative_binomial", {
  res = bi.dist.negative_binomial(total_count = 100, probs = 0.5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 88)
})

test_that("bi.dist.normal", {
  res = bi.dist.normal(
    loc = 0,
    scale = 2,
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, -0.41168428)
})

test_that("bi.dist.ordered_logistic", {
  res = bi.dist.ordered_logistic(
    predictor = c(0.2, 0.5, 0.8),
    cutpoints = c(-1.0, 0.0, 1.0),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(1, 1, 3))
})

test_that("bi.dist.pareto", {
  res = bi.dist.pareto(
    scale = c(0.2, 0.5, 0.8),
    alpha = c(-1.0, 0.5, 1.0),
    sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c( 0.11630858,  0.8140766 , 23.06902268))
})

test_that("bi.dist.poisson", {
  res = bi.dist.poisson(rate = c(0.2, 0.5, 0.8), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c( 0, 0, 1))
})

test_that("bi.dist.projected_normal", {
  res = bi.dist.projected_normal(concentration = c(1.0, 3.0, 2.0), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.17713475, 0.49410196, 0.85116774))
})


test_that("bi.dist.projected_normal", {
  res = bi.dist.projected_normal(concentration = c(1.0, 3.0, 2.0), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.17713475, 0.49410196, 0.85116774))
})

test_that("bi.dist.relaxed_bernoulli", {
  res = bi.dist.relaxed_bernoulli(temperature = 1, logits = 0.0, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.41845711))
})

test_that("bi.dist.right_truncated_distribution", {
  res = bi.dist.right_truncated_distribution(base_dist = bi.dist.normal(0,1, create_obj = TRUE), high = 10, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(-0.20584214))
})

test_that("bi.dist.soft_laplace", {
  res = bi.dist.soft_laplace(loc = 0, scale = 2, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(-0.51804666))
})

test_that("bi.dist.student_t", {
  res = bi.dist.student_t(df = 2, loc = 0, scale = 2, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(2.70136417))
})

test_that("bi.dist.truncated_cauchy", {
  res = bi.dist.truncated_cauchy(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.55196115))
})

test_that("bi.dist.truncated_distribution", {
  res = bi.dist.truncated_distribution(base_dist = bi.dist.normal(0,1, create_obj = TRUE), high = 0.7, low = 0.1, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.33487084))
})

test_that("bi.dist.truncated_normal", {
  res = bi.dist.truncated_normal(loc = 0, scale = 2, low = 0, high = 1.5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.58158364))
})

test_that("bi.dist.truncated_polya_gamma", {
  res = bi.dist.truncated_polya_gamma(batch_shape = c(), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.13129763))
})

test_that("bi.dist.two_sided_truncated_distribution", {
  res = bi.dist.two_sided_truncated_distribution(base_dist = bi.dist.normal(0,1, create_obj = TRUE), high = 0.5, low = 0.1, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.261847325))
})

test_that("bi.dist.uniform", {
  res = bi.dist.uniform(low = 0, high = 1.5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(0.62768567))
})

test_that("bi.dist.weibull", {
  res = bi.dist.weibull(scale = c(10, 10), concentration = c(1,1), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, c(5.42070555, 2.43723185))
})


test_that("bi.dist.wishart", {
  res = bi.dist.wishart(concentration = 5, scale_matrix = matrix(c(1,0,0,1), nrow = 2), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  lst <- list(
    c(5.81512786, -3.37817265),
    c(-3.37817265, 9.33345547)
  )
  expect_equal(r2,lst)
})

test_that("bi.dist.wishart_cholesky", {
  res = bi.dist.wishart_cholesky(concentration = 5, scale_matrix = matrix(c(1,0,0,1), nrow = 2), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  lst <- list(
    c(2.41145762,  0.),
    c(-1.4008841 ,  2.71495473)
  )
  expect_equal(r2,lst)
})

test_that("bi.dist.zero_inflated_distribution", {
  res = bi.dist.zero_inflated_distribution(base_dist = bi.dist.poisson(5, create_obj = TRUE), gate=0.3, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 4)
})

test_that("bi.dist.zero_inflated_negative_binomial", {
  res = bi.dist.zero_inflated_negative_binomial(mean = 2, concentration = 1, gate=0.3, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 1)
})

test_that("bi.dist.zero_inflated_poisson", {
  res = bi.dist.zero_inflated_poisson(gate=0.3, rate = 5, sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, 4)
})

test_that("bi.dist.zero_sum_normal", {
  res = bi.dist.zero_sum_normal(scale=0.3, event_shape = c(), sample = TRUE)
  r2 = reticulate::py_to_r(res$tolist())
  expect_equal(r2, -0.061752642)
})
}else{
  if(!test1){
    message("reticulate package is not available and required.")
  }
  if(!test2){
    message("Python is not available and required.")
  }
  if(!test3){
    message("BayesInference module is not available and required.")
  }
}


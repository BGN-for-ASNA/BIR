if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(testthat)
  library(BayesForge)

  draw_twice <- function(seed, f) {
    m <- importBF(platform = "cpu", rand_seed = seed, print_devices_found = FALSE)
    first <- reticulate::py_to_r(f()$tolist())
    m <- importBF(platform = "cpu", rand_seed = seed, print_devices_found = FALSE)
    second <- reticulate::py_to_r(f()$tolist())
    list(first, second)
  }

  test_that("same integer seed gives the same draws", {
    d <- draw_twice(42, function() bf.dist.normal(0, 1, sample = TRUE, shape = c(5L)))
    expect_equal(d[[1]], d[[2]])
  })

  test_that("different seeds give different draws", {
    a <- draw_twice(1, function() bf.dist.normal(0, 1, sample = TRUE, shape = c(5L)))[[1]]
    b <- draw_twice(2, function() bf.dist.normal(0, 1, sample = TRUE, shape = c(5L)))[[1]]
    expect_false(isTRUE(all.equal(a, b)))
  })

  test_that("consecutive draws in a session are not identical", {
    m <- importBF(platform = "cpu", rand_seed = 7, print_devices_found = FALSE)
    a <- reticulate::py_to_r(bf.dist.normal(0, 1, sample = TRUE)$tolist())
    b <- reticulate::py_to_r(bf.dist.normal(0, 1, sample = TRUE)$tolist())
    expect_false(isTRUE(all.equal(a, b)))
  })

  test_that("R seeds are passed as integers", {
    m <- importBF(platform = "cpu", rand_seed = 42, print_devices_found = FALSE)
    expect_equal(reticulate::py_to_r(m$get_seed()), 42L)
  })
}

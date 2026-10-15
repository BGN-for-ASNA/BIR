if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  # Tidy data in: an R data.frame with a factor; tidy draws out.
  set.seed(1)
  df <- data.frame(group = factor(rep(c("north", "south", "east"), each = 10),
                                  levels = c("south", "north", "east")))
  df$y <- c(south = 1, north = 2, east = 3)[as.character(df$group)] + rnorm(30, 0, 0.3)

  m <- importBF(platform = "cpu", rand_seed = 3, print_devices_found = FALSE)
  invisible(m$data(df))
  invisible(m$index("group"))
  m$data_to_model(list("index_group", "y"))
  model <- bf_model(function(index_group, y) {
    alpha <- bf.dist.normal(0, 5, shape = c(3L))
    s <- bf.dist.exponential(1)
    bf.dist.normal(alpha[index_group], s, obs = y)
  })
  m$fit(model, num_warmup = 300L, num_samples = 300L, num_chains = 2L, progress_bar = FALSE)

  test_that("a factor is indexed in its R level order", {
    codes <- m$df[["index_group"]]                  # converted to an R vector
    expect_identical(as.integer(codes), as.integer(df$group) - 1L)
    expect_identical(unlist(reticulate::py_to_r(m$index_map)$group, use.names = FALSE),
                     levels(df$group))
  })

  test_that("long draws hold exactly the wide draws, with factor labels", {
    wide <- bf_draws(m)
    long <- bf_draws(m, "long", labels = list(alpha = levels(df$group)))
    expect_identical(names(long), c(".chain", ".iteration", ".draw", ".variable", ".index", ".value"))
    expect_identical(nrow(long), nrow(wide) * (ncol(wide) - 3L))
    for (k in seq_along(levels(df$group))) {
      v <- long$.value[long$.variable == "alpha" & long$.index == levels(df$group)[k]]
      expect_identical(v, wide[[paste0("alpha[", k, "]")]])
    }
    expect_true(all(is.na(long$.index[long$.variable == "s"])))
    # the labels land on the right groups: posterior means near the true means
    means <- tapply(long$.value[long$.variable == "alpha"], long$.index[long$.variable == "alpha"], mean)
    expect_equal(as.vector(means[c("south", "north", "east")]), c(1, 2, 3), tolerance = 0.15)
  })

  test_that("labels are checked", {
    expect_error(bf_draws(m, "long", labels = list(beta = "a")), "no parameter called beta")
    expect_error(bf_draws(m, "long", labels = list(alpha = c("a", "b"))), "one label per element")
  })
}

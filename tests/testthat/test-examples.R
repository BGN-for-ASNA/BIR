# 2.6 -- every example of the distribution help pages runs. The help is
# generated from the Python docstrings (tools/build_r_docs.py); its examples
# are \donttest{}, which R CMD check does not run, so they are run here.
if (requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("BayesForge")) {
  library(BayesForge)

  test_that("every bf.dist.* help example runs", {
    db <- tools::Rd_db("BayesForge")
    pages <- grep("^bf\\.dist\\..*\\.Rd$", names(db), value = TRUE)
    expect_gt(length(pages), 70)
    failed <- character()
    for (p in pages) {
      ex <- tempfile(fileext = ".R")
      tools::Rd2ex(db[[p]], ex, commentDonttest = FALSE)
      if (!file.exists(ex)) next
      ok <- tryCatch({
        suppressMessages(suppressWarnings(capture.output(sys.source(ex, envir = new.env(parent = globalenv())))))
        TRUE
      }, error = function(e) FALSE)
      if (!ok) failed <- c(failed, p)
    }
    expect_identical(failed, character())
  })
}

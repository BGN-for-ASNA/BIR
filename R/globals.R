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


#' @title Convert BI fit to brmsfit
#' @description
#' Converts a BI (NumPyro MCMC) object into a compatible brmsfit object.
#' @param bi_fit The reticulate object returned by BI fitting.
#' @param formula The brms formula for the model. Defaults to `bi_fit$formula`.
#' @param data The data used for fitting. Defaults to `bi_fit$data`.
#' @param family The family used. Defaults to `bi_fit$family` or `gaussian()`.
#' @param par_map Optional list for custom naming: list(BI_name = brms_name).
#' @param ... Additional arguments passed to brms::brm(..., empty = TRUE).
#' @return A brmsfit object.
#' @import brms
#' @import rstan
#' @import reticulate
#' @import abind
#' @export
to_brms <- function(bi_fit, formula = NULL, data = NULL, family = NULL, par_map = NULL, ...) {
  # 0. Support default extraction if bi_fit has them
  if (is.null(formula) && !is.null(bi_fit$formula)) formula <- bi_fit$formula
  
  if (is.null(data)) {
      if (!is.null(bi_fit$data)) {
          data <- bi_fit$data
      } else if (!is.null(bi_fit$data_on_model)) {
          # Auto-convert BI data to R data frame
          message("Converting BI data_on_model to data.frame...")
          py_data <- bi_fit$data_on_model
          r_data <- reticulate::py_to_r(py_data)
          # Convert jax arrays to R arrays/vectors
          r_data <- lapply(r_data, function(x) {
              if (inherits(x, "python.builtin.object")) reticulate::py_to_r(x) else x
          })
          data <- as.data.frame(r_data)
      }
  }
  
  if (is.null(family)) {
    if (!is.null(bi_fit$family)) family <- bi_fit$family
    else family <- gaussian()
  }
  
  if (is.null(formula) || is.null(data)) {
    stop("Formula and data must be provided either as arguments or as properties of bi_fit.")
  }

  # 1. Extract samples and metadata
  samples_info <- bi_extract_samples(bi_fit)
  
  # 2. Create skeleton brmsfit
  brms_fit <- brms::brm(
    formula = formula,
    data = data,
    family = family,
    empty = TRUE,
    ...
  )

  # 3. Rename and map parameters
  # We try to match what brms expects for fixed effects
  sdata <- brms::make_standata(formula, data, family)
  fe_names <- attr(sdata$X, "dimnames")[[2]]
  
  samples_renamed <- samples_info$samples
  
  # Map Intercept
  if ("Intercept" %in% names(samples_renamed)) {
      samples_renamed$b_Intercept <- samples_renamed$Intercept
  } else if ("alpha" %in% names(samples_renamed)) {
      samples_renamed$b_Intercept <- samples_renamed$alpha
      samples_renamed$Intercept <- samples_renamed$alpha
  } else if ("a" %in% names(samples_renamed)) {
      # Custom 'a' for intercept
      samples_renamed$b_Intercept <- samples_renamed$a
      samples_renamed$Intercept <- samples_renamed$a
  }

  # Map noise/sigma
  if ("sigma" %in% names(samples_renamed)) {
      # Standard sigma
  } else if ("s" %in% names(samples_renamed)) {
      # Custom 's' for sigma
      samples_renamed$sigma <- samples_renamed$s
  }

  # Map slopes
  if (!is.null(fe_names)) {
      # Identify which parameter names are slopes (not Intercept)
      slope_fe <- fe_names[fe_names != "Intercept"]
      
      for (i in seq_along(slope_fe)) {
          fe_name <- slope_fe[i]
          
          # Try to find match: explicit name, b_<name>, or 'b' if single slope
          bi_name <- NULL
          if (fe_name %in% names(samples_renamed)) {
              bi_name <- fe_name
          } else if (paste0("b_", fe_name) %in% names(samples_renamed)) {
              bi_name <- paste0("b_", fe_name)
          } else if (length(slope_fe) == 1 && "b" %in% names(samples_renamed)) {
              # If there is only one slope in the formula AND the user chose 'b' as parameter name
              bi_name <- "b"
          } else if (!is.null(par_map) && (fe_name %in% par_map)) {
              bi_name <- names(par_map)[par_map == fe_name]
          }
          
          if (!is.null(bi_name)) {
              samples_renamed[[paste0("b_", fe_name)]] <- samples_renamed[[bi_name]]
          }
      }
  }
  
  # Final manual mapping
  if (!is.null(par_map)) {
    for (n in names(par_map)) {
      samples_renamed[[par_map[[n]]]] <- samples_renamed[[n]]
    }
  }

  # 4. Create stanfit object
  stan_fit <- bi_to_stanfit(
    samples_renamed,
    chains = samples_info$num_chains,
    iter = samples_info$num_samples,
    warmup = samples_info$num_warmup
  )
  
  # 5. Inject the stanfit into the brmsfit
  brms_fit$fit <- stan_fit
  
  return(brms_fit)
}

#' @keywords internal
bi_extract_samples <- function(bi_fit) {
  samples <- bi_fit$get_samples(group_by_chain = TRUE)
  samples_r <- reticulate::py_to_r(samples)
  
  extra <- tryCatch(bi_fit$get_extra_fields(group_by_chain = TRUE), error = function(e) NULL)
  if (!is.null(extra)) {
    extra_r <- reticulate::py_to_r(extra)
    if ("potential_energy" %in% names(extra_r)) {
      samples_r$lp__ <- -extra_r$potential_energy
    }
  }
  
  return(list(
    samples = samples_r,
    num_chains = as.integer(bi_fit$num_chains),
    num_samples = as.integer(bi_fit$num_samples),
    num_warmup = as.integer(bi_fit$num_warmup)
  ))
}

#' @keywords internal
bi_to_stanfit <- function(samples_list, chains, iter, warmup) {
  p_names <- names(samples_list)
  # Standardize lp__
  if ("lp__" %in% p_names) {
      p_names_sorted <- c(setdiff(p_names, "lp__"), "lp__")
  } else {
      p_names_sorted <- p_names
      samples_list$lp__ <- array(0, dim = c(chains, iter))
      p_names_sorted <- c(p_names_sorted, "lp__")
  }
  
  fnames_total <- c()
  samples_sim <- list()
  for (i in 1:chains) samples_sim[[i]] <- list()
  par_dims <- list()

  for (p in p_names_sorted) {
      val <- samples_list[[p]]
      d <- dim(val)
      if (length(d) <= 2) {
          fnames_total <- c(fnames_total, p)
          par_dims[[p]] <- integer(0)
          for (c in 1:chains) samples_sim[[c]][[p]] <- as.numeric(val[c,])
      } else {
          K_total <- prod(d[3:length(d)])
          par_dims[[p]] <- as.integer(d[3:length(d)])
          p_fnames <- paste0(p, "[", 1:K_total, "]")
          fnames_total <- c(fnames_total, p_fnames)
          for (c in 1:chains) {
              slice <- abind::asub(val, c, 1, drop = TRUE)
              flat_slice <- matrix(slice, nrow = d[2], ncol = K_total)
              for (k in 1:K_total) samples_sim[[c]][[p_fnames[k]]] <- as.numeric(flat_slice[,k])
          }
      }
  }

  fit <- new("stanfit", model_name = "BI_model")
  fit@model_pars <- setdiff(p_names_sorted, "lp__")
  fit@par_dims <- par_dims[fit@model_pars]
  fit@mode <- 0L
  fit@date <- date()
  
  stan_args <- list()
  for (c in 1:chains) {
      stan_args[[c]] <- list(chain_id = c, iter = iter, warmup = warmup, thin = 1L, method = "sampling", algorithm = "NUTS")
  }
  fit@stan_args <- stan_args
  
  # CRITICAL: rstan dimnames() looks for fnames_oi
  n_s <- length(samples_sim[[1]][[fnames_total[1]]])
  fit@sim <- list(
      samples = samples_sim, 
      iter = n_s, 
      warmup = 0, 
      chains = chains, 
      thin = 1L,
      n_save = rep(n_s, chains), 
      warmup2 = rep(0, chains),
      fnames_oi = fnames_total, # FOR dimnames()
      fnames_poi = fnames_total # FOR extraction
  )
  
  # Add sampler_params attribute to each chain
  for (c in 1:chains) {
      sp_list <- list()
      for (sp in c("accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__")) {
          sp_list[[sp]] <- rep(if (sp == "accept_stat__") 0.8 else 0, n_s)
      }
      attr(fit@sim$samples[[c]], "sampler_params") <- sp_list
  }

  fit@.MISC <- new.env()
  fit@.MISC$stan_has_run <- TRUE
  return(fit)
}

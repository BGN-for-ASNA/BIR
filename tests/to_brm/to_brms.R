library(brms)
library(reticulate)
library(rstan)
library(abind)

#' @title Convert BI fit to brmsfit
#' @description
#' Converts a BI (NumPyro MCMC) object into a compatible brmsfit object.
#' @param bi_fit The reticulate object returned by BI fitting.
#' @param formula The brms formula for the model.
#' @param data The data used for fitting.
#' @param family The family used (default: gaussian()).
#' @param par_map Optional list for custom naming: list(BI_name = brms_name).
#' @param ... Additional arguments passed to brms::brm(..., empty = TRUE).
#' @return A brmsfit object.
#' @export
to_brms <- function(bi_fit, formula, data, family = gaussian(), par_map = NULL, ...) {
  # 1. Extract samples and metadata
  samples_info <- bi_extract_samples(bi_fit)
  
  # 2. Rename parameters to brms conventions
  samples_renamed <- bi_rename_parameters(samples_info$samples, par_map)
  
  # 3. Create stanfit object
  stan_fit <- bi_to_stanfit(
    samples_renamed,
    chains = samples_info$num_chains,
    iter = samples_info$num_samples,
    warmup = samples_info$num_warmup
  )
  
  # 4. Create skeleton brmsfit
  brms_fit <- brms::brm(
    formula = formula,
    data = data,
    family = family,
    empty = TRUE,
    ...
  )
  
  # 5. Inject the stanfit into the brmsfit
  brms_fit$fit <- stan_fit
  
  return(brms_fit)
}

#' @keywords internal
bi_extract_samples <- function(bi_fit) {
  # Assuming bi_fit is a NumPyro MCMC object
  # mcmc.get_samples(group_by_chain=True) returns {name: array(chains, iter, ...)}
  samples <- bi_fit$get_samples(group_by_chain = TRUE)
  samples_r <- reticulate::py_to_r(samples)
  
  # Try to get log_probability (potential_energy)
  extra <- tryCatch(bi_fit$get_extra_fields(group_by_chain = TRUE), error = function(e) NULL)
  if (!is.null(extra)) {
    extra_r <- reticulate::py_to_r(extra)
    if ("potential_energy" %in% names(extra_r)) {
      # Stan uses lp__ = -potential_energy
      samples_r$lp__ <- -extra_r$potential_energy
    }
  }
  
  # Extract metadata
  num_chains <- bi_fit$num_chains
  num_samples <- bi_fit$num_samples
  num_warmup <- bi_fit$num_warmup
  
  return(list(
    samples = samples_r,
    num_chains = as.integer(num_chains),
    num_samples = as.integer(num_samples),
    num_warmup = as.integer(num_warmup)
  ))
}

#' @keywords internal
bi_rename_parameters <- function(samples, par_map = NULL) {
  # Default mapping for standard parameters
  # This can be expanded as we discover more conventions
  default_map <- list(
    "Intercept" = "b_Intercept",
    "sigma" = "sigma"
  )
  
  # Merge user map with defaults
  if (!is.null(par_map)) {
    for (n in names(par_map)) {
      default_map[[n]] <- par_map[[n]]
    }
  }
  
  new_samples <- list()
  for (n in names(samples)) {
    new_name <- n
    if (n %in% names(default_map)) {
      new_name <- default_map[[n]]
    }
    new_samples[[new_name]] <- samples[[n]]
  }
  
  return(new_samples)
}

#' @keywords internal
#' @keywords internal
bi_to_stanfit <- function(samples_list, chains, iter, warmup) {
  # Ensure abind is used for subsetting
  
  # 1. Parameter names and ordering
  p_names <- names(samples_list)
  if ("lp__" %in% p_names) {
    # Stan conventionally puts lp__ at the end of parameter lists sometimes, 
    # but for sim$samples it just needs to be consistent.
    p_names <- c(setdiff(p_names, "lp__"), "lp__")
  }
  
  # 2. Build the 'sim' structure
  # sim$samples is a list of lists: [[chain]][[parameter_vector]]
  # Note: rstan expects each parameter in the inner list to be a numeric vector of ALL samples.
  # If a parameter is a vector in the model, it should be flattened into separate names like p[1], p[2]...
  
  samples_sim <- list()
  fnames_total <- c()
  
  # First pass: identify all flattened parameter names
  # We'll use the first chain to determine names
  for (p in p_names) {
    val <- samples_list[[p]]
    # (chains, samples, ...)
    slice <- abind::asub(val, 1, 1, drop = TRUE)
    dims <- dim(slice)
    if (is.null(dims) || length(dims) == 1) {
      fnames_total <- c(fnames_total, p)
    } else if (length(dims) == 2) {
      # Vector parameter
      K <- dims[2]
      fnames_total <- c(fnames_total, paste0(p, "[", 1:K, "]"))
    } else {
      fnames_total <- c(fnames_total, p) # Fallback
    }
  }
  
  # Second pass: fill samples
  sampler_params <- c("accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__")
  fnames_total <- c(fnames_total, sampler_params)
  
  for (c in 1:chains) {
    chain_list <- list()
    for (p in p_names) {
      val <- samples_list[[p]]
      slice <- abind::asub(val, c, 1, drop = TRUE)
      dims <- dim(slice)
      
      if (is.null(dims) || length(dims) == 1) {
        chain_list[[p]] <- as.numeric(slice)
      } else if (length(dims) == 2) {
        K <- dims[2]
        for (i in 1:K) {
          chain_list[[paste0(p, "[", i, "]")]] <- as.numeric(slice[, i])
        }
      } else {
        chain_list[[p]] <- as.numeric(slice)
      }
    }
    
    # Determine actual number of samples in this slice
    n_samples_actual <- 0
    for (p in p_names) {
        v <- samples_list[[p]]
        s <- abind::asub(v, c, 1, drop = TRUE)
        if (is.null(dim(s))) {
            n_samples_actual <- length(s)
        } else {
            n_samples_actual <- dim(s)[1]
        }
        break # Just need it once
    }

    # Add dummy sampler parameters as an attribute
    sp_list <- list()
    for (sp in sampler_params) {
       if (sp == "accept_stat__") {
         sp_list[[sp]] <- rep(0.8, n_samples_actual)
       } else if (sp == "stepsize__") {
         sp_list[[sp]] <- rep(0.1, n_samples_actual)
       } else {
         sp_list[[sp]] <- rep(0, n_samples_actual)
       }
    }
    
    # Ensure columns match fnames_total and order
    this_chain <- chain_list[setdiff(fnames_total, sampler_params)]
    attr(this_chain, "sampler_params") <- sp_list
    samples_sim[[c]] <- this_chain
  }
  
  # Update fnames_total to NOT include sampler_params if they are only in the attribute
  # Actually, brms/rstan might want them in both or just attribute. 
  # But get_sampler_params only looks at the attribute.
  fnames_poi <- setdiff(fnames_total, sampler_params)
  
  # Determine effective iter/warmup based on actually available samples
  # If BI gave us 100 samples, and we say warmup=50, Stan expectations might be confused
  # safest: set warmup=0 and iter=n_samples_actual
  iter_final <- n_samples_actual
  warmup_final <- 0
  
  # 3. Initialize stanfit object
  fit <- new("stanfit", model_name = "BI_model")
  
  # model_pars are the base names (without indices)
  fit@model_pars <- setdiff(p_names, "lp__")
  
  # par_dims...
  par_dims <- list()
  for (p in fit@model_pars) {
     val <- samples_list[[p]]
     # (chains, samples, ...)
     slice <- abind::asub(val, 1, 1, drop = TRUE)
     d <- dim(slice)
     if (is.null(d) || length(d) == 1) {
       par_dims[[p]] <- integer(0) # scalar
     } else {
       par_dims[[p]] <- as.integer(d[2:length(d)]) # vector/matrix dims
     }
  }
  fit@par_dims <- par_dims
  fit@mode <- 0L # Sampling
  fit@date <- date()
  
  # Populate stan_args (important for brms summary)
  stan_args <- list()
  for (c in 1:chains) {
    stan_args[[c]] <- list(
      chain_id = c,
      iter = iter_final + warmup_final,
      warmup = warmup_final,
      thin = 1L,
      method = "sampling",
      algorithm = "NUTS",
      backend = "numpyro" # Custom tag
    )
  }
  fit@stan_args <- stan_args
  
  # Update sim slot
  fit@sim <- list(
    samples = samples_sim,
    iter = iter_final + warmup_final,
    warmup = warmup_final,
    chains = chains,
    thin = 1L,
    n_save = rep(iter_final + warmup_final, chains),
    warmup2 = rep(warmup_final, chains),
    fnames_poi = fnames_poi
  )
  
  fit@.MISC <- new.env()
  fit@.MISC$stan_has_run <- TRUE
  # rstan often needs this for as.array 
  
  return(fit)
}

# Parallel simulation runner: the R side of BayesForge.Parallel.simulate.
# Core splitting and seed derivation stay in Python (_plan) so R and Python
# runs agree exactly; scheduling (PSOCK) and results (data.frames) are R.

# The only place m$name differs from plain Python attribute access: `grid` and
# `run_simulations` are answered in R, everything else falls through to the
# Python object.
`$.BayesForge.Main.main.bf` <- function(x, name) {
  if (identical(name, "grid")) return(.bf_grid)
  if (identical(name, "run_simulations")) {
    # parent.frame() here is the caller of m$run_simulations(): `export` names are looked up there.
    return(function(...) .bf_run_simulations(..., .envir = parent.frame()))
  }
  NextMethod()
}

# Scenario grid: forwards to Python grid() and returns a data.frame whose
# columns are the parameters plus sim (0-based, as in Python) and rep.
.bf_grid <- function(reps = 1L, ...) {
  sim <- reticulate::import("BayesForge.Parallel.simulate")
  params <- list(...)
  # reticulate turns a multi-element vector into a numpy array, which Python
  # grid() treats as a single value; a list travels as a Python list.
  params[] <- lapply(params, function(v) if (length(v) > 1L) as.list(v) else v)
  g <- do.call(sim$grid, c(list(reps = as.integer(reps)), params))
  .bf_rbind_fill(lapply(g, .bf_as_row))
}

# Shape `scenarios` into what Python _normalise_scenarios accepts: an int, or
# an unnamed R list of named lists (reticulate sends a list of dicts).
.bf_scenarios_to_py <- function(scenarios) {
  if (is.numeric(scenarios) && length(scenarios) == 1L) {
    return(as.integer(scenarios))
  }
  if (is.data.frame(scenarios)) {
    # One named list per row; the outer list stays unnamed so reticulate
    # converts it to a Python list, not a dict.
    return(lapply(seq_len(nrow(scenarios)), function(i) {
      lapply(scenarios[i, , drop = FALSE], function(v) {
        if (is.factor(v)) v <- as.character(v)
        v[[1L]]  # every scenario value must be length 1
      })
    }))
  }
  scenarios
}

.bf_run_simulations <- function(fn, scenarios, workers = NULL, num_chains = NULL,
                                cores_per_worker = NULL, fn_kwargs = NULL,
                                export = NULL, seed = 0L,
                                on_error = c("record", "raise"), quiet = TRUE,
                                progress = TRUE,
                                backend = c("process", "serial"),
                                .envir = parent.frame()) {
  on_error <- match.arg(on_error)
  backend <- match.arg(backend)
  stopifnot(is.function(fn))
  if (!is.null(workers)) workers <- as.integer(workers)
  if (!is.null(num_chains)) num_chains <- as.integer(num_chains)
  if (!is.null(cores_per_worker)) cores_per_worker <- as.integer(cores_per_worker)

  sim <- reticulate::import("BayesForge.Parallel.simulate")
  # _plan normalises scenarios, derives seeds and splits cores in Python, so
  # both languages budget and seed identically.
  plan <- sim$`_plan`(.bf_scenarios_to_py(scenarios), workers = workers,
                      num_chains = num_chains,
                      cores_per_worker = cores_per_worker,
                      seed = as.integer(seed), quiet = quiet)
  # Warnings travel back as data; report them the R way.
  for (w in plan$warnings) warning(w, call. = FALSE)
  env <- plan$env
  # run_simulations budgets CPU cores; without this every worker's JAX probes for
  # CUDA at import (before importBF(platform=) runs) and prints a fallback warning
  # straight to the terminal. A JAX_PLATFORMS already set by the user wins.
  env$JAX_PLATFORMS <- Sys.getenv("JAX_PLATFORMS", "cpu")

  # One scenario, start to rows. Mirrors Python _run_one so the serial path is
  # a genuine debugging equivalent of the worker path.
  task <- function(scenario) {
    args <- c(fn_kwargs, scenario)
    # Python passes every key; R would fail with "unused argument" on extras.
    if (!("..." %in% names(formals(fn)))) {
      args <- args[names(args) %in% names(formals(fn))]
    }
    t0 <- proc.time()[["elapsed"]]
    # Worker stdout goes nowhere, so fn's cat/print/message output is captured
    # here and travels back with the result. tryCatch sits inside capture.output
    # so the lines printed before an error stay in the log; the handler takes
    # message() conditions, which would otherwise reach the worker's stderr.
    log <- character()
    ran <- withCallingHandlers(
      {
        # capture.output evaluates its argument here, so res1 lands in this frame.
        out <- utils::capture.output(res1 <- tryCatch(
          # Keep the scenario in the output even on failure, like Python's [{}].
          list(rows = .bf_sim_rows(do.call(fn, args)), error = NA_character_),
          error = function(e) list(rows = data.frame(row.names = 1L),
                                   error = conditionMessage(e))
        ))
        log <- c(log, out)
        res1
      },
      message = function(m) {
        log <<- c(log, sub("\n$", "", conditionMessage(m)))
        invokeRestart("muffleMessage")
      })
    seconds <- proc.time()[["elapsed"]] - t0
    meta <- .bf_as_row(c(scenario, list(
      worker = Sys.getpid(),
      elapsed_s = round(seconds, 3),
      error = ran$error
    )))
    # Meta first, result second: a result column of the same name wins and the
    # meta copy is dropped, as Python _run_one does.
    list(
      rows = lapply(seq_len(nrow(ran$rows)), function(i) {
        row <- ran$rows[i, , drop = FALSE]
        keep <- setdiff(names(meta), names(row))
        cbind(meta[, keep, drop = FALSE], row)
      }),
      log = log, sim = scenario$sim, error = ran$error, elapsed = seconds
    )
  }
  # A fresh env with parent globalenv() holds exactly what the worker needs,
  # so PSOCK ships these objects and workers do not need the installed package
  # to have the helpers.
  e <- new.env(parent = globalenv())
  e$fn <- fn
  e$fn_kwargs <- fn_kwargs
  e$.bf_sim_rows <- .bf_sim_rows
  e$.bf_rbind_fill <- .bf_rbind_fill
  e$.bf_as_row <- .bf_as_row
  # Re-home the helper copies too, or they would keep the package namespace as
  # their enclosure and resolve each other from the installed package.
  for (h in c(".bf_sim_rows", ".bf_rbind_fill", ".bf_as_row")) environment(e[[h]]) <- e
  environment(task) <- e

  # The settings a progress line is worth showing: everything Python added
  # (sim, rep, seed, num_chains) is bookkeeping, not a scenario value.
  n <- length(plan$scenarios)
  scenario_cols <- if (n)
    setdiff(names(plan$scenarios[[1L]]), c("sim", "rep", "seed", "num_chains"))
  else character()
  results <- vector("list", n)
  if (backend == "serial" || plan$workers == 1) {
    # Serial is the debugging path: stop at the first failure under raise.
    for (i in seq_len(n)) {
      scenario <- plan$scenarios[[i]]
      out <- task(scenario)
      if (progress) .bf_sim_report(out, "main", i, n, scenario_cols)
      if (on_error == "raise" && !is.na(out$error)) {
        stop(sprintf("simulation %d failed: %s", scenario$sim, out$error),
             call. = FALSE)
      }
      results[[i]] <- out
    }
  } else {
    cl <- parallel::makePSOCKcluster(plan$workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    # Before any Python starts in the worker: XLA reads its device count once, at backend start.
    set_env <- function(env) do.call(Sys.setenv, env)
    # globalenv() parent so the closure does not serialise the caller's frame,
    # which may hold the Python object m.
    environment(set_env) <- globalenv()
    parallel::clusterCall(cl, set_env, env)
    # One disjoint CPU block per worker, set before the JAX backend comes up (see _pin_worker).
    pin <- function(block, block_size) {
      suppressPackageStartupMessages(library(BayesForge))
      reticulate::import("BayesForge.Parallel.simulate")$`_pin_worker`(as.integer(block), as.integer(block_size))
      invisible(NULL)
    }
    environment(pin) <- globalenv()
    parallel::clusterApply(cl, seq_along(cl) - 1L, pin, block_size = plan$block_size)
    if (length(export)) parallel::clusterExport(cl, export, envir = .envir)
    if (progress) {
      # the cores split is optional in the plan; print what the plan reports
      hdr <- if (is.null(plan$cores_per_worker))
        sprintf("run_simulations: %d simulations on %d workers\n", n, length(cl))
      else
        sprintf("run_simulations: %d simulations on %d workers (%d cores each)\n",
                n, length(cl), plan$cores_per_worker)
      cat(hdr)
    }
    # future runs the tasks on the cluster set up above (only exported API, so
    # the package stays CRAN-clean). One future per simulation, at most one per
    # worker; poll and report each result as it arrives (dynamic dispatch).
    old_plan <- future::plan(future::cluster, workers = cl)
    on.exit(future::plan(old_plan), add = TRUE, after = FALSE)
    pids <- unlist(parallel::clusterCall(cl, Sys.getpid))
    submit <- function(j) {
      scenario <- plan$scenarios[[j]]
      future::future(list(pid = Sys.getpid(), value = task(scenario)),
                     globals = list(task = task, scenario = scenario),
                     seed = TRUE, stdout = FALSE)
    }
    first <- min(n, length(cl))
    running <- lapply(seq_len(first), submit)
    tags <- seq_len(first)
    nxt <- first + 1L
    done <- 0L
    while (length(running)) {
      ready <- which(vapply(running, future::resolved, logical(1)))
      if (!length(ready)) {
        Sys.sleep(0.05)
        next
      }
      k <- ready[[1L]]
      d <- tryCatch(future::value(running[[k]]), error = function(e) {
        stop("run_simulations: a worker process failed: ", conditionMessage(e),
             call. = FALSE)
      })
      j <- tags[[k]]
      running[[k]] <- NULL
      tags <- tags[-k]
      if (nxt <= n) {
        running[[length(running) + 1L]] <- submit(nxt)
        tags <- c(tags, nxt)
        nxt <- nxt + 1L
      }
      done <- done + 1L
      results[[j]] <- d$value
      if (progress) .bf_sim_report(d$value, match(d$pid, pids), done, n, scenario_cols)
    }
  }

  # Each task returns a list of one-row data.frames; flatten one level and
  # union the columns, since scenarios may return different shapes.
  rows <- lapply(results, function(r) r$rows)
  res <- .bf_rbind_fill(unlist(rows, recursive = FALSE))
  if ("sim" %in% names(res)) {
    res <- res[order(res$sim), , drop = FALSE]
  }
  row.names(res) <- NULL

  if (on_error == "raise") {
    bad <- which(!is.na(res$error))
    if (length(bad)) {
      # Workers finish out of order; report the lowest sim so the
      # failure does not depend on worker scheduling.
      i <- bad[which.min(res$sim[bad])]
      stop(sprintf("simulation %d failed: %s", res$sim[i], res$error[i]),
           call. = FALSE)
    }
  }
  bad <- which(!is.na(res$error))
  if (length(bad)) {
    i <- bad[which.min(res$sim[bad])]
    message(sprintf(
      "run_simulations: %d row(s) carry an error -- first (sim %d): %s -- see res[!is.na(res$error), ]",
      length(bad), res$sim[i], substr(res$error[i], 1L, 200L)
    ))
  }
  res
}

# One progress block per finished simulation: a header line, then whatever fn
# printed, indented under it. Called only when progress is TRUE.
.bf_sim_report <- function(r, node, done, total, scenario_cols) {
  # The scenario is not returned as a separate field; the first merged result
  # row carries its columns. At most 4 values fit on one terminal line.
  vals <- ""
  if (length(r$rows)) {
    row <- r$rows[[1L]]
    shown <- utils::head(intersect(scenario_cols, names(row)), 4L)
    if (length(shown))
      vals <- paste(shown,
                    vapply(shown, function(k) format(row[[k]], digits = 4), character(1)),
                    sep = "=", collapse = ", ")
  }
  status <- if (is.na(r$error)) "ok" else paste0("ERROR: ", substr(r$error, 1L, 80L))
  cat(sprintf("[%*d/%d] worker %s | sim %d | %s | %.1fs | %s\n",
              nchar(total), done, total, node, r$sim, vals, r$elapsed, status))
  log <- r$log
  while (length(log) && !nzchar(trimws(log[length(log)]))) log <- log[-length(log)]
  # 9 spaces line the log up under the header text.
  if (length(log)) cat(paste0("         ", log, "\n"), sep = "")
}

# Mirror of Python _normalise_result: whatever fn returned becomes a
# data.frame of result rows.
.bf_sim_rows <- function(ret) {
  if (inherits(ret, "BayesForge.Main.main.bf")) {
    stop("the simulation function returned the `m` object. A fitted model cannot cross a process boundary. Return summaries instead, e.g. m$summary() or a list of the numbers you need.", call. = FALSE)
  }
  if (inherits(ret, "python.builtin.object")) {
    # A dict/DataFrame return value arrives as Python; convert, then normalise.
    ret <- reticulate::py_to_r(ret)
  }
  if (is.null(ret)) {
    # No result is still a result: one row, no columns (Python's [{}]).
    return(data.frame(row.names = 1L))
  }
  if (is.list(ret) && !is.data.frame(ret) && !length(ret)) {
    # An empty list is no result, like NULL: keep the scenario row (Python's [{}]).
    return(data.frame(row.names = 1L))
  }
  if (is.data.frame(ret)) {
    row.names(ret) <- NULL
    return(ret)
  }
  if (is.list(ret) &&
      all(vapply(ret, function(el) is.list(el) && !is.null(names(el)), logical(1)))) {
    return(.bf_rbind_fill(lapply(ret, .bf_as_row)))
  }
  if ((is.list(ret) || is.atomic(ret)) && !is.null(names(ret))) {
    return(.bf_as_row(as.list(ret)))
  }
  data.frame(value = ret)
}

# A named list of length-1 values -> one-row data.frame.
.bf_as_row <- function(x) {
  # as.data.frame drops NULL elements; an explicit NA keeps the column.
  x[vapply(x, is.null, logical(1))] <- list(NA_character_)
  as.data.frame(x, stringsAsFactors = FALSE, check.names = FALSE)
}

# rbind data.frames with differing columns: union of names in first-seen
# order, missing columns filled with NA. Base R only (workers run this too).
.bf_rbind_fill <- function(dfs) {
  if (!length(dfs)) return(data.frame())
  nms <- unique(unlist(lapply(dfs, names)))
  filled <- lapply(dfs, function(d) {
    miss <- setdiff(nms, names(d))
    if (length(miss)) d[miss] <- NA
    d[, nms, drop = FALSE]
  })
  out <- do.call(rbind, filled)
  row.names(out) <- NULL
  out
}

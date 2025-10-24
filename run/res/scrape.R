load_quietly <- function(...) {
  suppressMessages(suppressWarnings(library(...)))
}

load_quietly(dplyr)
load_quietly(glue)
load_quietly(readr)
load_quietly(rlang)
load_quietly(stringr)


cmd_args <- commandArgs(trailingOnly = TRUE)
if (is_empty(cmd_args)) {
  abort("Please provide path to the experiment. ")
}

setwd(file.path(Sys.getenv("QTNH_DIR"), "run/res"))
out_dir <- file.path(Sys.getenv("QTNH_DIR"), "run/out")
exp_dir <- cmd_args[[1]]

data_dir <- file.path(out_dir, exp_dir)
files_out <- list.files(data_dir, "*.out")
data_table <- tibble()

rm_patterns = character(0)
if (length(length(cmd_args)) > 1) {
  rm_patterns <- cmd_args[2:length(cmd_args)]
}

for (file in files_out) {
  lines <- read_lines(file.path(data_dir, file))

  # Which line separates metadata and output
  line_exp <- str_which(lines, "^Experiment:")
  line_exe <- str_which(lines, "^Running.*:")
  line_start <- line_exe + 1
  line_end <- str_which(lines, "^Finished running") - 1

  # Match metadata in the form FIELD=value
  meta_lines <- str_extract(lines[1:(line_exe - 1)], "^[A-Z_0-9]+=.*")
  meta_lines <- na.omit(meta_lines)
  meta_names <- str_replace(meta_lines, "=.*", "")
  meta_vals  <- str_replace(meta_lines, ".*=", "")

  meta_data <- as.list(meta_vals)
  names(meta_data) <- meta_names

  # Extract other metadata
  if (length(line_exp) > 0) {
    exp <- str_split_1(lines[[line_exp]], ": ")[[2]]
    meta_data[["EXPERIMENT"]] <- exp
  }

  exe <- str_split_1(lines[[line_exe]], ":? ")[[2]]
  exe <- str_split_1(exe, "/")
  exe <- exe[[length(exe)]]
  meta_data[["EXECUTABLE"]] <- exe

  # Extract executable arguments
  exe_args <- str_split_1(lines[[line_exe]], ":? ")
  exe_args <- str_extract(exe_args, "^[0-9]+$")
  exe_args <- as.numeric(na.omit(exe_args))
  exe_names <- paste0("ARG", seq_along(exe_args))

  exe_data <- as.list(exe_args)
  names(exe_data) <- exe_names
  meta_data <- c(meta_data, exe_data)

  # Extract experiment output
  out <- lines[line_start:line_end]
  for (p in rm_patterns) {
    out <- out[str_which(out, p, negate = TRUE)]
  }

  out <- str_c(out, collapse = "\n")
  data <- c(meta_data, OUTPUT = out)

  data_table <- rbind(data_table, as_tibble(data))
}

write_csv(data_table, glue("raw/{exp_dir}.csv"))

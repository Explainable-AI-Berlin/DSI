library(tidyverse)
library(ggplot2)
library(here)
library(kableExtra)
library(patchwork)
library(ggvenn)

stitch_ci <- function(val, low, up){
  val = formatC(val, format = "f", digits = 2)
  low = formatC(low, format = "f", digits = 2)
  up = formatC(up, format = "f", digits = 2)
  out = str_c(val, " {\\scriptsize", "(", low, ", ", up, ")", "}")
  return(out)
}

stitch_minmax <- function(val, low, up){
  val = formatC(val, format = "f", digits = 2)
  low = formatC(low, format = "f", digits = 2)
  up = formatC(up, format = "f", digits = 2)
  out = str_c(val, " {\\tiny", low, "-", up, "}")
  return(out)
}

task_order <- c(
  "fruit_v_animal_3", "object_v_concept_3", "choose_first_of_3",
  "adjective_v_verb_3", "color_v_animal_3",  "choose_middle_of_3", 
  "capitalize", "present-past", "product-company",
  "country-capital", "sentiment", "english-french"
)

abstractive <- c(
  "capitalize",
  "country-capital",
  "english-french",
  "present-past",
  "sentiment",
  "product-company"
)
extractive <- c(
  "object_v_concept_3",
  "choose_first_of_3",
  "adjective_v_verb_3",
  "color_v_animal_3",
  "choose_middle_of_3" ,
  "fruit_v_animal_3"
)

models <- c("Qwen3-8B", "gemma-3-4b-it", "Llama-3.2-3B-Instruct")

model_names <- c(
  "Llama-3.2-3B-Instruct" ~ "Llama 3.2 (3B)",
  "gemma-3-4b-it" ~ "Gemma 3 (4B)",
  "Qwen3-8B" ~ "Qwen 3 (8B)"
)

sizes <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
loss_suffix <- "_log_likelihood"

fig_file <- function(name) here("figures", str_c(name, loss_suffix, ".pdf"))

load_results <- function(dir, models, suffix = ""){
  results <- 
    data.frame(model = models) %>% 
    mutate(
      data = map(models, ~read_csv(here("results", dir, str_c(.x, suffix, ".csv")))
      )) %>% 
    unnest(data)
  if("task" %in% names(results)){
    results <- mutate(
      results, 
      task = factor(task, level = task_order)
    )
  }
  return(results)
}

# main results (Table 1 + Fig. 3) -----------------------------------------

## load data
baseline <- load_results("baseline", models)
no_iterative <- load_results("test", models, "_robust")
not_robust <- load_results("test", models, "_not_robust")
iterative <- load_results("iterative", models, loss_suffix)
single_layer <- load_results("iterative", models, str_c("_single_layer", loss_suffix))

iterative$method = "DSI"
no_iterative$method = "no refinement"
not_robust$method = "no refinement &\nno gradient robustification"
single_layer$method = "DSI 1-layer"

res_combined <- bind_rows(iterative, no_iterative, not_robust)
res_combined_sl <- bind_rows(res_combined, single_layer)

fv_long <- read_csv("results_fv/results.csv") %>% 
  mutate(task = factor(task, level = task_order))

# Figure 3 + Appendix E
plot_accuracies <- function(results, baseline, fv, m, name, save){
  results <- results %>% filter(model == m, size %in% sizes)
  baseline <- baseline %>% filter(model == m)
  fv <- fv %>% filter(model == m, method == "bl_fv_zs")
  p <- results %>% 
    ggplot(aes(x = size, y = accuracy, color = method)) +
    geom_line() +
    geom_ribbon(
      data = results,
      aes(x = size, y = accuracy, fill = method, ymin = ci_lower, ymax = ci_upper),
      color = NA,
      alpha = .2) +
    facet_wrap(~task, nrow = 2) +
    scale_x_log10(breaks=sizes) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    geom_hline(
      data = baseline, 
      aes(yintercept = `0_shot_accuracy`, linetype = "0-shot baseline"), 
      alpha = .5
    ) +
    geom_hline(
      data = baseline, 
      aes(yintercept = n_shot_accuracy, linetype = "10-shot baseline"), 
      alpha = .5
    ) + 
    geom_hline(
      data = baseline,
      aes(yintercept = majority_accuracy, linetype = "majority class baseline"),
      alpha = .5
    ) + 
    geom_hline(
      data = fv,
      aes(yintercept = accuracy, color = "FV best layer"),
      alpha = .5
      ) +
    scale_color_manual(
      values = c(
        "DSI" = "black",
        "FV best layer" = "#2274A5",
        "DSI direct" = "#A72608",
        "no refinement" = "#ce9642",
        "no refinement &\nno gradient robustification" = "#3b7c70"),
      breaks=c(
        'DSI', 'DSI direct', 'no refinement',
        'no refinement &\nno gradient robustification', "FV best layer")
    ) +
    scale_fill_manual(
      values = c(
        "DSI" = "black",
        "DSI direct" = "#A72608",
        "no refinement" = "#ce9642",
        "no refinement &\nno gradient robustification" = "#3b7c70"),
      breaks=c(
        'DSI', 'DSI direct', 'no refinement',
        'no refinement &\nno gradient robustification')) +
    scale_linetype_manual(
      values = c(1,2,6)
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10),
      axis.title.x = element_text(size = 13, vjust = 0),
      axis.title.y = element_text(size = 13),
      legend.title=element_text(size=11), 
      legend.text=element_text(size=9)) +
    labs(colour = "Intervention", x = "# Neurons", y = "Accuracy", linetype = "Baselines") +
    guides(fill = "none", colour = guide_legend(order = 1), linetype = guide_legend(order=2))
  if(save){
    ggsave(
      fig_file(paste0(name, "_", m)),
      plot = p,
      width = 10,
      height = 4)
  }
  return(p)
}

plot_accuracies(res_combined, baseline, fv_long, models[1], "acc_task", TRUE)
plot_accuracies(res_combined, baseline, fv_long, models[2], "acc_task", TRUE)
plot_accuracies(res_combined, baseline, fv_long, models[3], "acc_task", TRUE)

# distributed -------------------------------------------------------------

res_combined_sl %>% 
  filter(method %in% c("DSI", "DSI 1-layer"), size %in% sizes) %>% 
  select(!c(size_check, ci_lower, ci_upper, n_samples, scale, nsamples)) %>% 
  pivot_wider(values_from = accuracy, names_from = method) %>% 
  mutate(
    accuracy_diff = DSI - `DSI 1-layer`,
    model = case_match(model, !!!model_names)) %>% 
  group_by(size, model) %>% 
  summarise(accuracy_diff = mean(accuracy_diff)) %>% 
  ggplot(aes(x = size, y = accuracy_diff, color = model)) +
  geom_line() +
  scale_x_log10(breaks=sizes) +
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 0.3)) +
  scale_color_manual(
    values = c(
      "Qwen 3 (8B)" ="black", "Gemma 3 (4B)" = "#ce9642", "Llama 3.2 (3B)" = "#3b7c70"),
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10),
    axis.title.x = element_text(size = 15, vjust = 0),
    axis.title.y = element_text(size = 15),
    legend.title=element_text(size=13), 
    legend.text=element_text(size=11)) +
  labs(colour = "Model", x = "# Neurons", y = "Avg. Accuracy \n Difference")

ggsave(here("figures", "single_layer.pdf"), width = 6, height = 2.5)

# Table 1 -----------------------------------------------------------------
res_summary <- res_combined %>% 
  group_by(model) %>%
  mutate(size = ifelse(size == max(size), "$d$ model", size)) %>% 
  filter(size %in% c(64, "$d$ model")) %>% 
  ungroup() %>% 
  mutate(task_set = ifelse(task %in% abstractive, "abstractive", "extractive")) %>% 
  select(model, method, task_set, accuracy, size) %>% 
  group_by(model, method, size, task_set) %>% 
  summarise(
    mean_acc = mean(accuracy),
    min_acc = min(accuracy),
    max_acc = max(accuracy)) %>%
  mutate(accuracy = stitch_minmax(mean_acc, min_acc, max_acc)) %>% 
  select(model, task_set, method, size, accuracy) %>% 
  mutate(
    model = case_match(
      model,
      !!!model_names),
    method = case_match(
      method,
      "DSI" ~ "DSI",
      "no refinement" ~ "no refinement",
      "no refinement &\nno gradient robustification" ~ "not robust")) %>%
  pivot_wider(names_from = "model", values_from = "accuracy") %>% 
  mutate(size = as.character(size))

bl_summary <- baseline %>% 
  mutate(
    task_set = ifelse(task %in% abstractive, "abstractive", "extractive")) %>% 
  rename(accuracy = `0_shot_accuracy`) %>% 
  select(model, task_set, accuracy) %>% 
  group_by(model, task_set) %>% 
  summarise(
    mean_acc = mean(accuracy),
    min_acc = min(accuracy),
    max_acc = max(accuracy)) %>% 
  mutate(accuracy = stitch_minmax(mean_acc, min_acc, max_acc)) %>% 
  select(model, task_set, accuracy) %>% 
  mutate(
    model = case_match(
      model,
      !!!model_names)
    ) %>%
  pivot_wider(names_from = "model", values_from = "accuracy") %>% 
  mutate(size = "$0$", method = "Baseline")

fv_summary <- fv_long %>% 
  mutate(
    model = case_match(
      model,
      !!!model_names)) %>% 
  group_by(model, method, task_set, size) %>% 
  summarise(
    mean_acc = mean(accuracy),
    min_acc = min(accuracy),
    max_acc = max(accuracy)) %>%
  mutate(accuracy = stitch_minmax(mean_acc, min_acc, max_acc)) %>% 
  select(model, task_set, method, size, accuracy) %>% 
  mutate(
    method = case_match(
      method,
      "zs_fv" ~ "FV",
      "bl_fv_zs" ~ "FV best l."
    )) %>%
  pivot_wider(names_from = "model", values_from = "accuracy") %>% 
  mutate(size = "$d$ model")

main_table <- bind_rows(res_summary, bl_summary, fv_summary)

options(knitr.kable.NA = '')
main_table %>% 
  pivot_wider(names_from = task_set, values_from = c(`Llama 3.2 (3B)`, `Qwen 3 (8B)`, `Gemma 3 (4B)`)) %>% 
  mutate(
    size = factor(size, levels = c("$0$", "64", "$d$ model")),
    method = case_match(
      method,
      "DSI" ~ "DSI",
      "no refinement" ~ "no iter.",
      "not robust" ~ "no rob.",
      "FV best l." ~ "FV best",
      "FV" ~ "FV",
      "Baseline" ~ "baseline"),
    method = factor(method, levels = c("baseline", "FV", "FV best", "no rob.", "no iter.", "DSI")),
    sc = ifelse(method %in% c("baseline", "FV best", "FV"), "bm", "im")) %>% 
  ungroup() %>% 
  arrange(sc, size, method) %>% 
  select(size, method, contains("abstractive"), contains("extractive")) %>% 
  rename_with(
    ~str_remove(., "_abstractive")) %>% 
  rename_with(
    ~str_remove(., "\\(\\d+B\\)")) %>% 
  rename_with(
    ~str_replace(., "_extractive", " ")) %>% 
  knitr::kable(
    format = "latex",
    booktabs = T,
    digits = 2,
    escape = FALSE,
    align = "r",
  ) %>% 
  add_header_above(c(" " = 2, "abstractive" = 3, "extractive" = 3))

# neuron effects (Fig. 2) -------------------------------------------------

interactions <- read_csv("results/interactions/interactions.csv")

fun_lab <- function(x) {
  n <- length(x)
  labels <- rep("", n)
  labels[1] <- x[1]
  labels[n] <- x[n]
  return(labels)
}

interactions %>% 
  group_by(ind_2, pair) %>% 
  filter(s1 == 0) %>% 
  ggplot(aes(x = s2, y = fval, color = factor(ind_2))) +
  geom_line() +
  theme_minimal(base_size = 19) +
  labs(x = "Intervention scale", y = "Output logit", color = "Neuron\nrank") +
  scale_color_manual(
    values = c("#898e9f","#3b7c70", "#ce9642", "#2274A5", "#A72608", "black"),)+
  theme(plot.margin = margin(0,0,0,0))

ggsave(here("figures", "single_neuron_effects.pdf"), width = 4, height = 3)

interactions %>%
  filter(task == "english-french") %>% 
  filter(pair < 4) %>%
  ggplot(aes(x = s1, y = s2, z = fval)) +
  geom_contour_filled(bins = 40) +
  theme_classic(base_size = 14) +
  coord_fixed() +
  scale_fill_viridis_d(guide = guide_colorsteps(direction="vertical", title.position="top"), labels = fun_lab,) +
  labs(x = "Scale neuron 1", y = "Scale neuron 2", fill = "Output\nlogit") +
  facet_wrap(~interaction(ind_1, ind_2, sep = "x"), nrow = 1) +
  theme(plot.margin = margin(0, 0, 0, 0))

ggsave(here("figures", "neuron_pair_effects.pdf"), width = 7.5, height = 2.2)

# neuron positions (Appendix G) -------------------------------------------

neurons_per_head <- load_results("/figures/neurons_per_head", models, loss_suffix)
neurons_per_head <- mutate(
  neurons_per_head,
  head = head+1,
  layer = layer+1)

neurons_per_head %>% 
  filter(n_neurons != 0, method == "iterative", size %in% sizes) %>% 
  mutate(
    model = case_match(
      model,
      !!!model_names
    )) %>% 
  group_by(model, size) %>%
  summarise(n_heads = n()/length(neurons_per_head$task %>% unique)) %>% 
  ggplot(aes(x = size, y = n_heads, color = model)) +
  geom_line() +
  scale_y_log10(breaks = sizes)+
  scale_x_log10(breaks = sizes) +
  theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank())+
  xlab("# Neurons") +
  ylab("# Model components") +
  scale_color_manual(values = c("#898e9f","#3b7c70", "#ce9642")) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
    legend.text=element_text(size=9),
    legend.position = "top") +
  labs(color = "")

ggsave(fig_file("components"), width = 5, height = 4)

neurons_per_head %>% 
  filter(n_neurons != 0, size == 64, model == models[1], method == "iterative") %>%
  ggplot() +
  geom_tile(aes(y = head, x = layer, fill = n_neurons)) +
  theme_minimal(base_size = 14) +
  scale_fill_viridis_c(breaks = c(1, 19)) +
  scale_y_continuous(breaks = c(1, 8, 16, 24, 32), limits = c(0, 33)) +
  scale_x_continuous(breaks = c(1, 9, 18, 27, 36), limits = c(0, 37)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  facet_wrap(~task) +
  labs(x = "Layer", y = "Head", fill = "# Neurons")

ggsave(fig_file("Qwen3_neuron_layers_heads_bytask"), width = 10, height = 6)

neurons_per_head %>% 
  filter(n_neurons != 0, size == 64, model == models[2], method == "iterative") %>%
  ggplot() +
  geom_tile(aes(y = head, x = layer, fill = n_neurons)) +
  theme_minimal(base_size = 14) +
  scale_fill_viridis_c(breaks = c(1, 21)) +
  scale_y_continuous(breaks = c(1, 8), limits = c(0, 9)) +
  scale_x_continuous(breaks = c(1, 17, 34), limits = c(0, 35)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  facet_wrap(~task) +
  labs(x = "Layer", y = "Head", fill = "# Neurons")

ggsave(fig_file("gemma3_neuron_layers_heads_bytask"), width = 10, height = 6)

neurons_per_head %>% 
  filter(n_neurons != 0, size == 64, model == models[3], method == "iterative") %>%
  ggplot() +
  geom_tile(aes(y = head, x = layer, fill = n_neurons)) +
  theme_minimal(base_size = 14) +
  scale_fill_viridis_c(breaks = c(1, 16)) +
  scale_y_continuous(breaks = c(1, 8, 16, 24), limits = c(0, 25)) +
  scale_x_continuous(breaks = c(1, 14, 28), limits = c(0, 29)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  facet_wrap(~task) +
  labs(x = "Layer", y = "Head", fill = "# Neurons")

ggsave(fig_file("llama32_neuron_layers_heads_bytask"), width = 10, height = 6)


# Iterative result details (Appendix F) -----------------------------------

iterative_summary <- load_results("iterative", models, str_c("_summary", loss_suffix))

iterative_summary %>% 
  filter(size %in% sizes) %>% 
  mutate(
    mask_overlap = mask_overlap/size,
    task = factor(task, levels = task_order),
    model = case_match(
      model,
      !!!model_names
    )) %>% 
  ggplot(aes(x = size, y = mask_overlap, color = model)) +
  facet_wrap(~task, nrow = 2) +
  geom_line()+
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(
        axis.text.y = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10),
        axis.title.x = element_text(size = 13, vjust = 0),
        axis.title.y = element_text(size = 13),
        legend.title=element_text(size=11), 
        legend.text=element_text(size=9)) +
  facet_wrap(~task, nrow = 2) +
  scale_x_log10(breaks=sizes) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  scale_color_manual(values = c("#898e9f","#3b7c70", "#ce9642", "#898e9f","#2274A5", "#A72608","black", "orange", "green")) +
  scale_fill_manual(values = c("#898e9f","#2274A5", "#A72608", "black", "#2274A5")) +
  labs(colour = "", x = "# Neurons", y = "Neuron set overlap") +
  guides(fill = "none")

ggsave(fig_file("mask_overlap"), width = 10, height = 4)

intervention_strength <- load_results("/figures/intervention_strength", models, loss_suffix)

fv_norms <- read_csv("results_fv/norms.csv") %>% 
  mutate(
    model = case_match(
      model,
      !!!model_names
    )
  )

intervention_strength %>% 
  filter(size %in% sizes) %>% 
  mutate(
    task = factor(task, levels = task_order),
    model = case_match(
      model,
      !!!model_names
    ),
    method = case_match(
      method,
      "first_order" ~ "first order",
      "iterative" ~ "iterative"
    )) %>% 
  ggplot(aes(x = size, y = total_abs, color = method)) +
  geom_line()+
  geom_hline(data = fv_norms, aes(yintercept = l1_norm, color = "function vector"), linetype = 8) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(axis.text.y = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10),
        axis.title.x = element_text(size = 13, vjust = 0),
        axis.title.y = element_text(size = 13),
        legend.title=element_text(size=11), 
        legend.text=element_text(size=9)) +
  facet_wrap(model ~ task) +
  scale_x_log10(breaks=sizes) +
  scale_y_continuous(trans = "log10") + # breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  scale_color_manual(values = c("#898e9f","black","#3b7c70", "#ce9642", "#898e9f","#2274A5", "#A72608","black", "orange", "green")) +
  scale_fill_manual(values = c("#898e9f","#2274A5", "#A72608", "black", "#2274A5")) +
  labs(colour = "", x = "# Neurons", y = "Total absolute intervention") +
  guides(fill = "none")

ggsave(fig_file("total_intervention"), width = 10, height = 15)


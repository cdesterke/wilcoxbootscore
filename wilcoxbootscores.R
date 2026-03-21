library(randomForest)
data(iris)

library(dplyr)
set.seed(123)

iris_aug <- iris %>%
  mutate(
    Noise1 = rnorm(n()),   # bruit gaussien pur
    Noise2 = 1             # colonne constante
  ) %>%
  select(Noise1, Noise2, everything())   # <-- les mettre au début





X <- iris_aug[, 1:6]
y <- iris_aug$Species

X <- iris_aug[, 1:6]   # sélection initiale

# retirer les colonnes à variance nulle
X <- X[, sapply(X, function(col) var(col) > 0)]

## version long table

library(dplyr)
library(tidyr)



B <- 500  # nombre de bootstraps

# Table longue pour stocker tous les résultats
bootstrap_long <- data.frame(
  bootstrap = integer(),
  variable = character(),
  class = character(),
  score = numeric(),
  stringsAsFactors = FALSE
)

set.seed(123)

wilcox_effect_size <- function(x, g) {
  test <- wilcox.test(x[g], x[!g], exact = FALSE)
  z <- qnorm(test$p.value / 2) * sign(test$statistic - (length(x[g]) * length(x[!g]) / 2))
  r <- z / sqrt(length(x))
  return(r)
}

for (b in 1:B) {
  idx <- sample(1:nrow(X), replace = TRUE)
  Xb <- X[idx, ]
  yb <- y[idx]
  
  for (j in vars) {
    for (cls in classes) {
      g <- (yb == cls)
      
      # Wilcoxon effect size
      r <- wilcox_effect_size(Xb[[j]], g)
      
      # Correction du signe basée sur les moyennes réelles
      mean_diff <- mean(Xb[g, j]) - mean(Xb[!g, j])
      r <- sign(mean_diff) * abs(r)
      
      # Ajout en long format
      bootstrap_long <- rbind(
        bootstrap_long,
        data.frame(
          bootstrap = b,
          variable = j,
          class = cls,
          score = r
        )
      )
    }
  }
}


library(ggplot2)

ggplot(bootstrap_long, aes(x = class, y = score, fill = class)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.15, outlier.alpha = 0.2) +
  facet_wrap(~ variable, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Distribution of bootstrap Wilcoxon scores",
    x = "Classes",
    y = "Predictive scores"
  )


## detection des variables non productives
library(dplyr)

IC <- bootstrap_long %>%
  group_by(variable, class) %>%
  summarise(
    q25 = quantile(score, 0.25),
    q75 = quantile(score, 0.75),
    .groups = "drop"
  )

non_predictive_vars <- IC %>%
  group_by(variable) %>%
  summarise(
    overlap = max(q25) < min(q75)
  ) %>%
  filter(overlap == TRUE) %>%
  pull(variable)





X <- X[, !(colnames(X) %in% non_predictive_vars)]




## forme matrix

classes <- levels(y)
vars <- colnames(X)

B <- 500

score_mat <- matrix(0, nrow = length(vars), ncol = length(classes),
                    dimnames = list(vars, classes))

set.seed(123)

wilcox_effect_size <- function(x, g) {
  test <- wilcox.test(x[g], x[!g], exact = FALSE)
  z <- qnorm(test$p.value / 2) * sign(test$statistic - (length(x[g]) * length(x[!g]) / 2))
  r <- z / sqrt(length(x))
  return(r)
}

for (j in vars) {
  scores_j <- matrix(0, nrow = B, ncol = length(classes))
  
  for (b in 1:B) {
    idx <- sample(1:nrow(X), replace = TRUE)
    Xb <- X[idx, ]
    yb <- y[idx]
    
    for (k in seq_along(classes)) {
      cls <- classes[k]
      g <- (yb == cls)
      
      # Wilcoxon effect size
      r <- wilcox_effect_size(Xb[[j]], g)
      
      # Correction du signe basée sur les moyennes réelles
      mean_diff <- mean(Xb[g, j]) - mean(Xb[!g, j])
      r <- sign(mean_diff) * abs(r)
      
      scores_j[b, k] <- r
    }
  }
  
  score_mat[j, ] <- apply(scores_j, 2, median)
}

score_mat


importance <- rowSums(abs(score_mat))
importance

var_order <- names(sort(importance, decreasing = TRUE))
var_order

library(tidyverse)

# importance globale
importance <- rowSums(abs(score_mat))

# ordre décroissant (plus prédictives en premier)
var_order <- names(sort(importance, decreasing = TRUE))

# passage en long format
score_long <- as.data.frame(score_mat) %>%
  rownames_to_column("variable") %>%
  pivot_longer(cols = -variable, names_to = "class", values_to = "score")

# heatmap avec variables les plus prédictives EN HAUT
ggplot(score_long, aes(
  x = class,
  y = factor(variable, levels = rev(var_order)),   # <-- inversion ici
  fill = score
)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", score)), size = 4) +
  scale_fill_gradient2(
    low = "#4575b4", mid = "white", high = "#d73027",
    midpoint = 0, name = "Score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Heatmap of Wilcox predictive scores",
    x = "Classes",
    y = "Variables"
  )








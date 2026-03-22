library(glmnetr)

## simulation metastase 10 genes

set.seed(123)

n <- 250
p <- 10
p_signal <- 4

library(MASS)

Sigma <- matrix(0.3, p, p)
diag(Sigma) <- 1

X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
colnames(X) <- paste0("G", 1:p)

beta_true <- c(1.5, -1.2, 1.0, -0.8, rep(0, p - p_signal))

linpred <- X %*% beta_true
prob <- 1 / (1 + exp(-linpred))

y <- rbinom(n, 1, prob)

df <- data.frame(y = y, X)

ls("package:glmnetr", all.names = TRUE)

library(glmnet)
library(pROC)
library(dplyr)
library(randomForest)
library(xgboost)
library(nnet)

Xmat <- as.matrix(df[, -1])
yvec <- df$y


eval_glmnet <- function(alpha, name) {
  cvfit <- cv.glmnet(Xmat, yvec, family = "binomial", alpha = alpha)
  preds <- predict(cvfit, Xmat, s = "lambda.min", type = "response")
  auc <- pROC::auc(yvec, preds)
  tibble(model = name, alpha = alpha, auc = as.numeric(auc))
}


eval_xgb <- function() {
  dtrain <- xgb.DMatrix(data = Xmat, label = yvec)
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1
  )
  xgbfit <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
  preds <- predict(xgbfit, Xmat)
  auc <- pROC::auc(yvec, preds)
  tibble(model = "XGBoost", auc = as.numeric(auc))
}


eval_ann <- function() {
  ann <- nnet(y ~ ., data = df, size = 5, linout = FALSE, skip = TRUE, maxit = 500, trace = FALSE)
  preds <- predict(ann, df[, -1])
  auc <- pROC::auc(yvec, preds)
  tibble(model = "ANN (nnet)", auc = as.numeric(auc))
}

eval_logit <- function() {
  logit <- glm(y ~ ., data = df, family = binomial)
  preds <- predict(logit, type = "response")
  auc <- pROC::auc(yvec, preds)
  tibble(model = "Logistic Regression", auc = as.numeric(auc))
}

eval_rf <- function() {
  rf <- randomForest(x = Xmat, y = as.factor(yvec), ntree = 500)
  preds <- predict(rf, Xmat, type = "prob")[,2]
  auc <- pROC::auc(yvec, preds)
  tibble(model = "Random Forest", auc = as.numeric(auc))
}




results <- bind_rows(
  eval_glmnet(0,   "Ridge"),
  eval_glmnet(0.5, "Elastic Net"),
  eval_glmnet(1,   "Lasso"),
  eval_rf(),
  eval_xgb(),
  eval_ann(),
  eval_logit()
)

results %>% arrange(desc(auc))







results <- bind_rows(
  eval_glmnet(0,   "Ridge"),
  eval_glmnet(0.5, "Elastic Net"),
  eval_glmnet(1,   "Lasso")
)

results

## compare model graphes 

##multiroc
# glmnet models
ridge_fit  <- cv.glmnet(Xmat, yvec, family="binomial", alpha=0)
enet_fit   <- cv.glmnet(Xmat, yvec, family="binomial", alpha=0.5)
lasso_fit  <- cv.glmnet(Xmat, yvec, family="binomial", alpha=1)

# Random Forest
rf <- randomForest(x = Xmat, y = as.factor(yvec), ntree = 500)

# XGBoost
dtrain <- xgb.DMatrix(data = Xmat, label = yvec)
xgbfit <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1
  ),
  data = dtrain,
  nrounds = 200,
  verbose = 0
)
preds_xgb <- predict(xgbfit, dtrain)

# ANN
ann <- nnet(y ~ ., data = df, size = 5, linout = FALSE, skip = TRUE,
            maxit = 500, trace = FALSE)

# Logistic regression
logit <- glm(y ~ ., data = df, family = binomial)

roc_list <- list(
  Ridge = roc(yvec, as.numeric(predict(ridge_fit, Xmat, s="lambda.min", type="response"))),
  ElasticNet = roc(yvec, as.numeric(predict(enet_fit, Xmat, s="lambda.min", type="response"))),
  Lasso = roc(yvec, as.numeric(predict(lasso_fit, Xmat, s="lambda.min", type="response"))),
  RandomForest = roc(yvec, predict(rf, Xmat, type="prob")[,2]),
  XGBoost = roc(yvec, preds_xgb),
  ANN = roc(yvec, as.numeric(predict(ann, df[, -1]))),
  Logistic = roc(yvec, as.numeric(predict(logit, type="response")))
)

cols <- c(
  Ridge="#1f77b4",
  ElasticNet="#ff7f0e",
  Lasso="#2ca02c",
  RandomForest="#d62728",
  XGBoost="#9467bd",
  ANN="#8c564b",
  Logistic="#17becf"
)

# Plot initial
plot(roc_list[[1]], col=cols[1], lwd=2,
     main="Comparaison des modèles – Courbes ROC")

# Add others
i <- 1
for (name in names(roc_list)) {
  plot(roc_list[[name]], col=cols[i], lwd=2, add=TRUE)
  i <- i + 1
}

legend("bottomright", legend=names(roc_list),
       col=cols, lwd=2, cex=0.8)


## barplot AUC
cols <- c(
  Ridge="#1f77b4",
  ElasticNet="#ff7f0e",
  Lasso="#2ca02c",
  RandomForest="#d62728",
  XGBoost="#9467bd",
  ANN="#8c564b",
  Logistic="#17becf"
)

library(dplyr)




library(ggplot2)
library(dplyr)

results_plot <- results %>%
  mutate(model = recode(model,
    "Random Forest" = "RandomForest",
    "Logistic Regression" = "Logistic"
  ))

library(ggplot2)

ggplot(results_plot, aes(x = reorder(model, auc), y = auc, fill = model)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", auc)),
            hjust = -0.1, size = 4.5) +
  coord_flip(clip = "off") +
  scale_fill_manual(values = cols) +
  labs(
    title = "Comparaison des modèles – AUC",
    x = "Modèle",
    y = "AUC"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.margin = margin(10, 30, 10, 10)
  ) +
  ylim(0, max(results_plot$auc) + 0.05)











## XGB

set.seed(123)

idx <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))
train <- df[idx, ]
test  <- df[-idx, ]

Xtrain <- as.matrix(train[, -1])
ytrain <- train$y

Xtest  <- as.matrix(test[, -1])
ytest  <- test$y

## grille hyper parameters

grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(2, 3, 4),
  subsample = c(0.7, 0.9, 1),
  colsample_bytree = c(0.7, 1),
  nrounds = c(200, 400)
)


## fonction de CV interne XGB k-fold
library(xgboost)
library(pROC)
library(dplyr)

eval_xgb_cv <- function(params) {
  
  dtrain <- xgb.DMatrix(data = Xtrain, label = ytrain)
  
  cv <- xgb.cv(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = params$eta,
      max_depth = params$max_depth,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree
    ),
    data = dtrain,
    nrounds = params$nrounds,
    nfold = 5,
    verbose = 0,
    early_stopping_rounds = 20
  )
  
  tibble(
    auc = max(cv$evaluation_log$test_auc_mean),
    best_iter = cv$best_iteration
  )
}


## boucle tuning
library(tidyverse)
results_xgb <- grid %>%
  rowwise() %>%
  mutate(
    res = list(eval_xgb_cv(pick(everything())))
  ) %>%
  unnest(res) %>%
  arrange(desc(auc))



best <- results_xgb %>% slice(1)
best

## visualisation
## heatmap auc

library(ggplot2)
library(dplyr)

heat_xgb <- results_xgb %>%
  group_by(eta, max_depth) %>%
  slice_max(auc, n = 1, with_ties = FALSE) %>%
  ungroup()

library(ggplot2)

ggplot(heat_xgb,
       aes(x = factor(max_depth),
           y = factor(eta),
           fill = auc)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.3f", auc)),
            color = "white", size = 5, fontface = "bold") +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "XGBoost Tuning – AUC (5-fold CV)",
    x = "max_depth",
    y = "eta (learning rate)",
    fill = "AUC"
  ) +
  theme_minimal(base_size = 14)

## scatter plot multidim

ggplot(results_xgb,
       aes(x = eta,
           y = auc,
           color = factor(max_depth),
           size = subsample)) +
  geom_point(alpha = 0.8) +
  scale_color_viridis_d() +
  labs(
    title = "XGBoost Hyperparameter Tuning",
    x = "eta",
    y = "AUC (CV)",
    color = "max_depth",
    size = "subsample"
  ) +
  theme_minimal(base_size = 14)

## courbe auc nround meilleur model

best <- results_xgb %>% slice(1)

dtrain <- xgb.DMatrix(data = Xtrain, label = ytrain)

cv_best <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best$eta,
    max_depth = best$max_depth,
    subsample = best$subsample,
    colsample_bytree = best$colsample_bytree
  ),
  data = dtrain,
  nrounds = best$nrounds,
  nfold = 5,
  verbose = 0
)

df_cv <- cv_best$evaluation_log

ggplot(df_cv, aes(x = iter, y = test_auc_mean)) +
  geom_line(color = "#0072B2", linewidth = 1.2) +
  geom_vline(xintercept = cv_best$best_iteration, linetype = "dashed") +
  labs(
    title = "XGBoost CV – Courbe AUC vs nrounds",
    x = "nrounds",
    y = "AUC (test fold mean)"
  ) +
  theme_minimal(base_size = 14)









## entrainement du meilleur model

# Courbe CV
cv <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best$eta,
    max_depth = best$max_depth,
    subsample = best$subsample,
    colsample_bytree = best$colsample_bytree
  ),
  data = xgb.DMatrix(Xtrain, label=ytrain),
  nrounds = best$nrounds,
  nfold = 5,
  verbose = 0
)

df_cv <- cv$evaluation_log

# Best iteration
best_iter_cv <- df_cv$iter[which.max(df_cv$test_auc_mean)]
best_auc_cv  <- max(df_cv$test_auc_mean)

library(ggplot2)

ggplot(df_cv, aes(x = iter, y = test_auc_mean)) +
  geom_line(linewidth = 1.2, color="#9467bd") +
  geom_point(color="#9467bd") +
  
  # Ligne verticale au best iteration
  geom_vline(xintercept = best_iter_cv, linetype = "dashed", color="black") +
  
  # Point du maximum
  geom_point(aes(x = best_iter_cv, y = best_auc_cv),
             color="red", size=4) +
  
  # Label texte
  annotate("text",
           x = best_iter_cv,
           y = best_auc_cv,
           label = paste0("Best nrounds = ", best_iter_cv,
                          "\nAUC = ", sprintf("%.3f", best_auc_cv)),
           hjust = -0.1, vjust = -0.5,
           size = 5, fontface = "bold") +
  
  labs(
    title = "Courbe de gradient – AUC vs nrounds (XGBoost)",
    x = "nrounds",
    y = "AUC (CV)"
  ) +
  theme_minimal(base_size = 14)









dtrain <- xgb.DMatrix(data = Xtrain, label = ytrain)

best_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best$eta,
    max_depth = best$max_depth,
    subsample = best$subsample,
    colsample_bytree = best$colsample_bytree
  ),
  data = dtrain,
  nrounds = best_iter_cv,  
  verbose = 0
)

##best_iter_cv

dtest <- xgb.DMatrix(data = Xtest)
preds <- predict(best_model, dtest)
auc_test <- pROC::auc(ytest, preds)
auc_test

## evaluler le surapprentissage
best_auc_cv
preds_train <- predict(best_model, dtrain)
pROC::auc(ytrain, preds_train)
preds_test <- predict(best_model, dtest)
pROC::auc(ytest, preds_test)




## courbe roc auc
library(pROC)

dtest <- xgb.DMatrix(Xtest)
preds <- predict(best_model, dtest)

roc_xgb <- roc(ytest, preds)

plot(roc_xgb, col="#9467bd", lwd=3,
     main="ROC – XGBoost (best model)")
text(0.6, 0.2, labels = paste0("AUC = ", round(auc(roc_xgb), 3)),
     col="#9467bd", cex=1.4)

## calibration
library(ggplot2)
library(dplyr)

cal <- tibble(
  pred = preds,
  obs = ytest
) %>%
  mutate(bin = ntile(pred, 10)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred),
    mean_obs = mean(obs)
  )

ggplot(cal, aes(x = mean_pred, y = mean_obs)) +
  geom_point(size = 3, color="#9467bd") +
  geom_line(linewidth = 1.2, color="#9467bd") +
  geom_abline(slope = 1, intercept = 0, linetype="dashed") +
  labs(
    title = "Courbe de calibration – XGBoost",
    x = "Probabilité prédite",
    y = "Probabilité observée"
  ) +
  theme_minimal(base_size = 14)

## courbe de gain cumulé
library(gains)

g <- gains(actual = ytest, predicted = preds)

plot(g$depth, g$cume.pct.of.total, type="l", lwd=3, col="#9467bd",
     xlab="Proportion d'échantillons",
     ylab="Proportion de positifs capturés",
     main="Courbe de gain cumulatif – XGBoost")


## lift curve
plot(g$depth, g$lift, type="l", lwd=3, col="#9467bd",
     xlab="Proportion d'échantillons",
     ylab="Lift",
     main="Lift curve – XGBoost")


## courbe de gradients nrounds AUC

cv <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best$eta,
    max_depth = best$max_depth,
    subsample = best$subsample,
    colsample_bytree = best$colsample_bytree
  ),
  data = xgb.DMatrix(Xtrain, label=ytrain),
  nrounds = best$nrounds,
  nfold = 5,
  verbose = 0
)

df_cv <- cv$evaluation_log

ggplot(df_cv, aes(x = iter, y = test_auc_mean)) +
  geom_line(linewidth = 1.2, color="#9467bd") +
  geom_point(color="#9467bd") +
  labs(
    title = "Courbe de gradient – AUC vs nrounds",
    x = "nrounds",
    y = "AUC (CV)"
  ) +
  theme_minimal(base_size = 14)


## importances
imp <- xgb.importance(model = best_model)

imp <- imp %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))

ggplot(imp, aes(x = Feature, y = Gain, fill = Gain)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", Gain)),
            hjust = -0.1, size = 4.5) +
  coord_flip() +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Importance des variables – XGBoost (Gain)",
    x = "Variables",
    y = "Importance (Gain)"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  ylim(0, max(imp$Gain) * 1.15)


## shap
library(SHAPforxgboost)

# Calcul des valeurs SHAP
shap <- shap.values(
  xgb_model = best_model,
  X_train   = Xtrain
)

# Mise en forme longue pour le summary plot
shap_long <- shap.prep(
  shap_contrib = shap$shap_score,
  X_train      = Xtrain
)

# Summary plot (beeswarm)
shap.plot.summary(shap_long)


## shap barplot
shap_imp <- data.frame(
  Feature = names(shap$mean_shap_score),
  Importance = shap$mean_shap_score
) %>%
  arrange(desc(Importance)) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))

ggplot(shap_imp, aes(x = Feature, y = Importance, fill = Importance)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", Importance)),
            hjust = -0.1, size = 4.5) +
  coord_flip() +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Importance SHAP – XGBoost",
    x = "Variables",
    y = "Importance SHAP"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")


## shap dependance top features
library(SHAPforxgboost)

# Calcul des valeurs SHAP
shap <- shap.values(
  xgb_model = best_model,
  X_train   = Xtrain
)

# Top features (par importance SHAP moyenne)
top_features <- names(sort(shap$mean_shap_score, decreasing = TRUE))[1:6]
top_features


shap_long <- shap.prep(
  shap_contrib = shap$shap_score,
  X_train      = Xtrain
)


library(ggplot2)

plots_dep <- lapply(top_features, function(feat) {
  shap.plot.dependence(
    data_long = shap_long,
    x = feat,
    y = feat,
    color_feature = feat
  ) +
    ggtitle(paste("SHAP Dependence –", feat))
})

library(patchwork)

p_dep_multi <- wrap_plots(plots_dep, ncol = 3) +
  plot_annotation(
    title = "SHAP Dependence Plots – Top Features",
    theme = theme(plot.title = element_text(size = 18, face = "bold"))
  )

p_dep_multi



## extraction des lasso coefficients

library(glmnet)

# Matrice des top features
X_lasso <- as.matrix(Xtrain[, top_features])

# LASSO avec CV
cvfit <- cv.glmnet(
  x = X_lasso,
  y = ytrain,
  family = "binomial",
  alpha = 1
)

# Coefficients du modèle linéaire
coef_lasso <- coef(cvfit, s = "lambda.min")
coef_lasso

## score sur le test
Xtest_lasso <- as.matrix(Xtest[, top_features])
score_lasso <- predict(cvfit, newx = Xtest_lasso, s = "lambda.min", type = "response")

library(pROC)
auc_lasso <- auc(ytest, score_lasso)
auc_lasso

###

library(glmnet)

X_lasso   <- as.matrix(Xtrain[, top_features])
Xtest_las <- as.matrix(Xtest[,  top_features])
y_lasso   <- as.numeric(ytrain)  # 0/1

set.seed(123)

cvfit <- cv.glmnet(
  x      = X_lasso,
  y      = y_lasso,
  family = "binomial",
  alpha  = 1,          # LASSO
  nfolds = 10
)

cvfit$lambda.min
cvfit$cvm[cvfit$lambda == cvfit$lambda.min]

coef_lasso <- coef(cvfit, s = "lambda.min")
coef_lasso


coef_df <- data.frame(
  Feature = coef_lasso@Dimnames[[1]],
  Coef    = as.numeric(coef_lasso)
)

coef_df <- coef_df[coef_df$Coef != 0, ]  # on garde seulement les non nuls
coef_df

score_lasso <- predict(
  cvfit,
  newx = Xtest_las,
  s    = "lambda.min",
  type = "response"   # probabilité
)

score_lasso <- as.numeric(score_lasso)


library(pROC)

auc_lasso <- auc(ytest, score_lasso)
auc_lasso

beta0 <- coef_df$Coef[coef_df$Feature == "(Intercept)"]
betas <- coef_df$Coef[coef_df$Feature != "(Intercept)"]
names(betas) <- coef_df$Feature[coef_df$Feature != "(Intercept)"]

# Exemple de calcul manuel du score pour Xtest
score_manual <- as.numeric(beta0 + Xtest_las %*% betas)
prob_manual  <- 1 / (1 + exp(-score_manual))











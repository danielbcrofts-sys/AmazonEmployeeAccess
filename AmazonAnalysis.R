.libPaths("/yunity/dcrofts0/R/x86_64-pc-linux-gnu-library/4.5")

# ------- LIBRARIES  -------


library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(ggplot2)
library(kknn)
library(discrim)
library(themis)
library(embed)
library(kernlab)


# ------- DATA  -------

set.seed(123)
dat_train <- vroom("train.csv")
dat_test <- vroom("test.csv")
# glimpse(dat_train)
# glimpse(dat_test)


# ------- EDA -------

# # Plot 1
# ggplot(dat_train, aes(x = factor(ACTION), fill = factor(ACTION))) +
#   geom_bar() +
#   labs(
#     title = "Distribution of Access Granted vs. Denied",
#     x = "Action (0 = Denied, 1 = Granted)",
#     y = "Count"
#   ) +
#   scale_fill_manual(values = c("red", "darkgreen"), guide = FALSE)
# 
# 
# #Plot 2
# dat_train %>%
#   group_by(RESOURCE) %>%
#   summarize(approval_rate = mean(ACTION)) %>%
#   ggplot(aes(x = approval_rate)) +
#   geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
#   labs(
#     title = "Distribution of Approval Rates Across Resources",
#     x = "Approval Rate (per Resource)",
#     y = "Number of Resources"
#   )



# ------- DUMMY ENCODING AND NUM COLUMNS FOR HW1 -------

dat_train <- dat_train %>%
  mutate(ACTION = as.factor(ACTION))

amazon_recipe <- recipe(ACTION ~ ., data = dat_train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

amazon_prep <- prep(amazon_recipe)
amazon_baked <- bake(amazon_prep, new_data = dat_train)


# ------- LOGISTIC REGRESSION -------

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_workflow <- workflow() %>%
  add_model(logRegModel) %>%
  add_recipe(amazon_recipe)

logReg_fit <- fit(logReg_workflow, data = dat_train)

amazon_predictions <- predict(
  logReg_fit,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = amazon_predictions$.pred_1
)

write.csv(submission, "logreg_submission.csv", row.names = FALSE)


# ------- LOGISTIC PENALIZED REGRESSION -------

tuning_grid <- grid_regular(
  penalty(range = c(-6, 0)),  # slightly wider search
  mixture(range = c(0, 1)),   # full range from ridge to lasso
  levels = 4
)

folds <- vfold_cv(dat_train, v = 3)

CV_results <- logReg_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )
bestTune <- select_best(CV_results, metric = "roc_auc")

final_wf <- logReg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = dat_train)

amazon_predictions <- predict(
  final_wf,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = amazon_predictions$.pred_1
)

vroom::vroom_write(submission, "pen_logreg_submission.csv", delim = ',')



# ------- BINARY RANDOM FORESTS -------
amazon_recipe <- recipe(ACTION ~ ., data = dat_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_interact(terms = ~ starts_with("RESOURCE"):starts_with("ROLE") +
                       starts_with("MGR_ID"):starts_with("ROLE")) 

rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 200 
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(amazon_recipe)

rf_grid <- grid_regular(
  mtry(range = c(15, 45)), 
  min_n(range = c(3, 8)),
  levels = 3 
)

rf_folds <- vfold_cv(dat_train, v = 3) 

rf_results <- rf_workflow %>%
  tune_grid(
    resamples = rf_folds,
    grid = rf_grid,
    metrics = metric_set(roc_auc)
  )

best_rf <- select_best(rf_results, metric = "roc_auc")

final_rf_wf <- rf_workflow %>%
  finalize_workflow(best_rf) %>%
  fit(data = dat_train)

rf_predictions <- predict(
  final_rf_wf,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = rf_predictions$.pred_1
)

vroom::vroom_write(submission, "rf_submission.csv", delim = ",")


# ------- KNN -------

dat_train_small <- dat_train %>% sample_n(3000)

amazon_recipe_knn <- recipe(ACTION ~ ., data = dat_train_small) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(amazon_recipe_knn) %>%
  add_model(knn_model)

knn_grid <- tibble(neighbors = c(3, 7, 11))
knn_folds <- vfold_cv(dat_train_small, v = 3)

knn_results <- knn_wf %>%
  tune_grid(
    resamples = knn_folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

best_k <- select_best(knn_results, metric = "roc_auc")

final_knn_wf <- knn_wf %>%
  finalize_workflow(best_k) %>%
  fit(data = dat_train_small)

knn_predictions <- predict(
  final_knn_wf,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = knn_predictions$.pred_1
)

vroom::vroom_write(submission, "knn_submission.csv", delim = ",")


# ------- NAIVE BAYES -------

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>% 
  add_model(nb_model)

nb_grid <- grid_regular(
  Laplace(range = c(0, 2)),
  smoothness(range = c(0, 2)),
  levels = 3
)

folds <- vfold_cv(dat_train, v = 5, strata = ACTION)

nb_results <- tune_grid(
  nb_wf,
  resamples = folds,        
  grid = nb_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_nb <- select_best(nb_results, metric = "roc_auc")

final_nb <- finalize_workflow(nb_wf, best_nb)
nb_fit <- fit(final_nb, data = dat_train)

nb_predictions <- predict(nb_fit, new_data = dat_test, type = "class")

submission <- tibble(
  Id = 1:nrow(dat_test),
  Action = nb_predictions$.pred_class
)

vroom::vroom_write(submission, "naive_bayes_submission.csv", delim = ",")


# ------- NEURAL NETWORK -------

# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# reticulate::install_python()
# keras::install_keras()


# ------- PCA -------

## PCA PENALIZED LOGREG
amazon_recipe_pca <- amazon_recipe %>%
  step_normalize(all_predictors()) %>%             
  step_pca(all_predictors(), threshold = 0.9)

logReg_workflow_pca <- workflow() %>%
  add_recipe(amazon_recipe_pca) %>%
  add_model(
    logistic_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      set_engine("glmnet") %>%
      set_mode("classification")
  )

tuning_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(),
  levels = 5
)

folds <- vfold_cv(dat_train, v = 5)

CV_results_pca <- logReg_workflow_pca %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

bestTune_pca <- select_best(CV_results_pca, metric = "roc_auc")

final_logReg_pca <- logReg_workflow_pca %>%
  finalize_workflow(bestTune_pca) %>%
  fit(data = dat_train)

amazon_predictions_pca <- predict(
  final_logReg_pca,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = amazon_predictions_pca$.pred_1
)

vroom_write(submission, "logreg_pca_submission.csv", delim = ",")


## PCA RF
rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 100
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(amazon_recipe)

rf_grid <- grid_regular(
  mtry(range = c(10, 60)),
  min_n(range = c(2, 10)),
  levels = 3
)
rf_folds <- vfold_cv(dat_train, v = 5)

amazon_recipe_pca <- amazon_recipe %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9)

rf_workflow_pca <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(amazon_recipe_pca)

rf_results_pca <- rf_workflow_pca %>%
  tune_grid(
    resamples = rf_folds,
    grid = rf_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

best_rf_pca <- select_best(rf_results_pca, metric = "roc_auc")

final_rf_wf_pca <- rf_workflow_pca %>%
  finalize_workflow(best_rf_pca) %>%
  fit(data = dat_train)

rf_predictions_pca <- predict(
  final_rf_wf_pca,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = rf_predictions_pca$.pred_1
)

vroom_write(submission, "rf_pca_submission.csv", delim = ",")


## PCA KNN
dat_train_small <- dat_train %>% sample_n(3000)

amazon_recipe_knn_pca <- recipe(ACTION ~ ., data = dat_train_small) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9)

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf_pca <- workflow() %>%
  add_recipe(amazon_recipe_knn_pca) %>%
  add_model(knn_model)

knn_grid <- tibble(neighbors = c(3, 7, 11))
knn_folds <- vfold_cv(dat_train_small, v = 3)

knn_results_pca <- knn_wf_pca %>%
  tune_grid(
    resamples = knn_folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

best_k_pca <- select_best(knn_results_pca, metric = "roc_auc")

final_knn_wf_pca <- knn_wf_pca %>%
  finalize_workflow(best_k_pca) %>%
  fit(data = dat_train_small)

knn_predictions_pca <- predict(
  final_knn_wf_pca,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = knn_predictions_pca$.pred_1
)

vroom_write(submission, "knn_pca_submission.csv", delim = ",")


# ------- SMOTE IMBALANCED DATA -------

amazon_recipe_smote <- recipe(ACTION ~ ., data = dat_train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_smote(ACTION, neighbors = 5)

## Smote LogReg

tuning_grid <- grid_regular(
  penalty(range = c(-3, -1)),
  mixture(),
  levels = 2
)

folds <- vfold_cv(dat_train, v = 2)

logReg_workflow_smote <- workflow() %>%
  add_recipe(amazon_recipe_smote) %>%
  add_model(
    logistic_reg(penalty = tune(), mixture = tune()) %>%
      set_engine("glmnet") %>%
      set_mode("classification")
  )

CV_results_smote <- logReg_workflow_smote %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

bestTune_smote <- select_best(CV_results_smote, metric = "roc_auc")

final_logReg_smote <- logReg_workflow_smote %>%
  finalize_workflow(bestTune_smote) %>%
  fit(data = dat_train)

amazon_predictions_smote <- predict(
  final_logReg_smote,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = amazon_predictions_smote$.pred_1
)

vroom::vroom_write(submission, "logreg_smote_submission.csv", delim = ",")

## Smote Pen LogReg

pen_logreg_workflow_smote <- workflow() %>%
  add_recipe(amazon_recipe_smote) %>%
  add_model(
    logistic_reg(penalty = tune(), mixture = tune()) %>%
      set_engine("glmnet") %>%
      set_mode("classification")
  )

pen_logreg_results_smote <- pen_logreg_workflow_smote %>%
  tune_grid(
    resamples = folds,          # reuse your v = 2 folds
    grid = tuning_grid,         # reuse same 2x2 grid
    metrics = metric_set(roc_auc, accuracy)
  )

best_pen_logreg_smote <- select_best(pen_logreg_results_smote, metric = "roc_auc")

final_pen_logreg_smote <- pen_logreg_workflow_smote %>%
  finalize_workflow(best_pen_logreg_smote) %>%
  fit(data = dat_train)

pen_logreg_predictions_smote <- predict(
  final_pen_logreg_smote,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = pen_logreg_predictions_smote$.pred_1
)

vroom::vroom_write(submission, "pen_logreg_smote_submission.csv", delim = ",")

## Smote RF

rf_mod_smote <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_grid_smote <- grid_regular(
  mtry(range = c(10, 60)),
  min_n(range = c(2, 10)),
  levels = 5
)

rf_folds_smote <- vfold_cv(dat_train, v = 5)

rf_workflow_smote <- workflow() %>%
  add_recipe(amazon_recipe_smote) %>%
  add_model(rf_mod_smote)

rf_results_smote <- rf_workflow_smote %>%
  tune_grid(
    resamples = rf_folds_smote,
    grid = rf_grid_smote,
    metrics = metric_set(roc_auc, accuracy)
  )

best_rf_smote <- select_best(rf_results_smote, metric = "roc_auc")

final_rf_smote <- rf_workflow_smote %>%
  finalize_workflow(best_rf_smote) %>%
  fit(data = dat_train)

rf_predictions_smote <- predict(
  final_rf_smote,
  new_data = dat_test,
  type = "prob"
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = rf_predictions_smote$.pred_1
)

vroom::vroom_write(submission, "rf_smote_submission.csv", delim = ",")




# ------- SVMs -------

amazon_recipe_svm <- recipe(ACTION ~ ., data = dat_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.99) %>%
  step_downsample(ACTION)



# Linear SVM
svm_linear <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear_wf <- workflow() %>%
  add_recipe(amazon_recipe_svm) %>%
  add_model(svm_linear)

svm_linear_fit <- fit(svm_linear_wf, dat_train)

svm_linear_preds <- predict(svm_linear_fit, new_data = dat_test, type = "prob")

submission_linear <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = svm_linear_preds$.pred_1
)

vroom::vroom_write(submission_linear, "svm_linear_submission.csv", delim = ",")


# Polynomial SVM
svm_poly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- workflow() %>%
  add_recipe(amazon_recipe_svm) %>%
  add_model(svm_poly)

svm_poly_fit <- fit(svm_poly_wf, dat_train)

svm_poly_preds <- predict(svm_poly_fit, new_data = dat_test, type = "prob")

submission_poly <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = svm_poly_preds$.pred_1
)

vroom::vroom_write(submission_poly, "svm_poly_submission.csv", delim = ",")


# Radial SVM
svm_radial <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_radial_wf <- workflow() %>%
  add_recipe(amazon_recipe_svm) %>%
  add_model(svm_radial)

svm_radial_fit <- fit(svm_radial_wf, dat_train)

svm_radial_preds <- predict(svm_radial_fit, new_data = dat_test, type = "prob")

submission_radial <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = svm_radial_preds$.pred_1
)

vroom::vroom_write(submission_radial, "svm_radial_submission.csv", delim = ",")


rf <- vroom("rf_submission.csv", delim = ',')
log <- vroom("logreg_submission.csv", delim = ',')

ensemble <- tibble(
  Id = rf$Id,
  ACTION = (rf$ACTION + log$ACTION) / 2   # simple average of probabilities
)

vroom_write(ensemble, "ensemble_submission.csv", delim = ",")


.libPaths("/yunity/dcrofts0/R/x86_64-pc-linux-gnu-library/4.5")


library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(ggplot2)
library(kknn)
library(discrim)


## PASSWORD


set.seed(123)
dat_train <- vroom("train.csv")
dat_test <- vroom("test.csv")
# glimpse(dat_train)
# glimpse(dat_test)


### EDA

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


### DUMMY ENCODING AND NUM COLUMNS FOR HW1
dat_train <- dat_train %>%
  mutate(ACTION = as.factor(ACTION))

amazon_recipe <- recipe(ACTION ~ ., data = dat_train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


amazon_prep <- prep(amazon_recipe)
amazon_baked <- bake(amazon_prep, new_data = dat_train)

# ncol(amazon_baked)



### LOGISTIC REGRESSION

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_workflow <- workflow() %>%
  add_model(logRegModel) %>%
  add_recipe(amazon_recipe)

logReg_fit <- fit(logReg_workflow, data = dat_train)

amazon_predictions <- predict(
  logReg_fit,
  new_data = dat_test,
  type = "prob"   # "prob" gives predicted probabilities for both 0 and 1
)

submission <- tibble(
  Id = 1:nrow(dat_test),
  ACTION = amazon_predictions$.pred_1
)

write.csv(submission, "amazon_submission.csv", row.names = FALSE)


### LOGISTIC PENALIZED REGRESSION

tuning_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(),
  levels = 5
)

# CV

folds <- vfold_cv(dat_train, v = 5)


CV_results <- logReg_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

# ROC
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

vroom::vroom_write(submission, "submission.csv")

bestTune %>%
  dplyr::select(penalty, mixture)
bestTune




### BINARY RANDOM FORESTS


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

# grid of tuning values
rf_grid <- grid_regular(
  mtry(range = c(10, 60)),
  min_n(range = c(2, 10)),
  levels = 3
)
# cross-validation
rf_folds <- vfold_cv(dat_train, v = 5)

# Tune the model
rf_results <- rf_workflow %>%
  tune_grid(
    resamples = rf_folds,
    grid = rf_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

# best tuning parameters
best_rf <- select_best(rf_results, metric = "roc_auc")


final_rf_wf <- rf_workflow %>%
  finalize_workflow(best_rf) %>%
  fit(data = dat_train)

# predict on test dat
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


 

### KNN



dat_train_small <- dat_train %>% sample_n(3000)

amazon_recipe_knn <- recipe(ACTION ~ ., data = dat_train_small) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

# Define KNN model and workflow
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(amazon_recipe_knn) %>%
  add_model(knn_model)

# Very small tuning grid and few folds to speed up
knn_grid <- tibble(neighbors = c(3, 7, 11))
knn_folds <- vfold_cv(dat_train_small, v = 3)

# Tune quickly
knn_results <- knn_wf %>%
  tune_grid(
    resamples = knn_folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

best_k <- select_best(knn_results, metric = "roc_auc")

# Fit final model on the reduced training set
final_knn_wf <- knn_wf %>%
  finalize_workflow(best_k) %>%
  fit(data = dat_train_small)

# Predict on test data
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





### NAIVE BAYES

nb_model <- 
  naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# workflow
nb_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>% 
  add_model(nb_model)

# Cv and tune
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

# fit on training data
final_nb <- finalize_workflow(nb_wf, best_nb)
nb_fit <- fit(final_nb, data = dat_train)

# predict
nb_predictions <- predict(nb_fit, new_data = dat_test, type = "class")

submission <- tibble(
  Id = 1:nrow(dat_test),
  Action = nb_predictions$.pred_class
)

vroom::vroom_write(submission, "naive_bayes_submission.csv", delim = ",")




### NEURAL NETWORK
install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
keras::install_keras()






### PCA

##PCA PENALIZED LOGREG
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

# CV

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


##PCA RF
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

# grid of tuning values
rf_grid <- grid_regular(
  mtry(range = c(10, 60)),
  min_n(range = c(2, 10)),
  levels = 3
)
# cross-validation
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

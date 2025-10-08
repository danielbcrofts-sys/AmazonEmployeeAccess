library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(ggplot2)

dat_train <- vroom("train.csv")
dat_test <- vroom("test.csv")
glimpse(dat_train)
glimpse(dat_test)


### EDA

# Plot 1
ggplot(dat_train, aes(x = factor(ACTION), fill = factor(ACTION))) +
  geom_bar() +
  labs(
    title = "Distribution of Access Granted vs. Denied",
    x = "Action (0 = Denied, 1 = Granted)",
    y = "Count"
  ) +
  scale_fill_manual(values = c("#E57373", "#81C784"), guide = FALSE)


#Plot 2
dat_train %>%
  group_by(RESOURCE) %>%
  summarize(approval_rate = mean(ACTION)) %>%
  ggplot(aes(x = approval_rate)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Approval Rates Across Resources",
    x = "Approval Rate (per Resource)",
    y = "Number of Resources"
  )


### DUMMY ENCODING AND NUM COLUMNS FOR HW1
amazon_recipe <- recipe(ACTION ~ ., data = dat_train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%             # make all predictors factors
  step_other(all_nominal_predictors(), threshold = 0.001) %>%   # combine rare categories
  step_dummy(all_nominal_predictors())                           # dummy encode

amazon_prep <- prep(amazon_recipe)
amazon_baked <- bake(amazon_prep, new_data = dat_train)

ncol(amazon_baked)

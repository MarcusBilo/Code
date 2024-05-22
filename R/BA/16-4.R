# Entweder mit CTRL + ENTER von oben nach unten durchgehen
# Oder mit CTRL + SHIFT + S alles auf einmal machen
  
# CLEAR CONSOLE & MEMORY
cat("\014")
rm(list=ls())

# packages to load and install if needed
packages <- c("caret", "ranger", "glmnet", "dplyr", "tidyr", "randomForest", "ggplot2", "gridExtra")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# import data from current working directory
data_in <- paste0(getwd(), "/")
data <- read.csv(paste0(data_in,"used-cars_2cities_prep.csv"), stringsAsFactors = TRUE)

# print all columns with missing data, as well as amount of missing data
# columns_with_missing <- colnames(data)[colSums(is.na(data)) > 0]
# for (col in columns_with_missing) {
#   num_missing <- sum(is.na(data[col]))
#   cat("Column:", col, "- Missing values:", num_missing, "\n")
# }
# cat("\n")

# here we drop all NA data
# however imputation or some other form of handling would probably
# be better as we drop 418 out of the original 549 rows due to NA values
data <- na.omit(data)

# print summary
cat("Summary for Price - full data, before splitting:\n")
print(summary(data$price))
cat("\n")

# set seed
set.seed(123)

# train test split
partition <- createDataPartition(y = seq_len(nrow(data)), p = 0.3, list = FALSE)
data_30 <- data[partition, ]
data_70 <- data[-partition, ]

# rf and lasso models
ctrl <- trainControl(method = "cv", number = 5)
rf_model <- train(
  price ~ .,
  data = data_70,
  method = "rf",
  trControl = ctrl
)
lasso_model <- train(
  price ~ .,
  data = data_70,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 1, lambda = 0.1)
)

# RMSE train
# rf_rmse_train <- sqrt(mean((predict(rf_model, data_70) - data_70$price)^2))
# lasso_rmse_train <- sqrt(mean((predict(lasso_model, data_70) - data_70$price)^2))
# print(paste("Train - Random Forest RMSE:", rf_rmse_train))
# print(paste("Train - LASSO RMSE:", lasso_rmse_train))

# RMSE test
rf_rmse_test <- sqrt(mean((predict(rf_model, data_30) - data_30$price)^2))
lasso_rmse_test <- sqrt(mean((predict(lasso_model, data_30) - data_30$price)^2))
print(paste("Test - RF RMSE:", rf_rmse_test))
print(paste("Test - LASSO RMSE:", lasso_rmse_test))
cat("\n")

# -----------------------------------------------------------------------------

# Calculate predictions for test set using random forest and lasso models
# Create data frame for random forest predictions
# Create data frame for lasso predictions
# Reshape the data frames to long format
# Filter the data to include only actual and predicted values for the same index
# Create plots
# Arrange plots stacked on top of each other

# Test: ---------------------------------------
rf_pred_test <- predict(rf_model, data_30)
lasso_pred_test <- predict(lasso_model, data_30)

plot_data_rf <- data.frame(
  Index = 1:nrow(data_30),
  Actual = data_30$price,
  Predicted = rf_pred_test
)

plot_data_lasso <- data.frame(
  Index = 1:nrow(data_30),
  Actual = data_30$price,
  Predicted = lasso_pred_test
)

plot_data_rf$Difference <- abs(plot_data_rf$Actual - plot_data_rf$Predicted)
plot_data_lasso$Difference <- abs(plot_data_lasso$Actual - plot_data_lasso$Predicted)

# Summary for Random Forest
rf_summary <- summary(plot_data_rf$Difference)
cat("Summary for Actual vs Predicted - Absolute Difference - Test - Random Forest Model:\n")
print(rf_summary)

# Summary for Lasso
lasso_summary <- summary(plot_data_lasso$Difference)
cat("\nSummary for Actual vs Predicted - Absolute Difference - Test - Lasso Model:\n")
print(lasso_summary)


plot_data_long_rf <- tidyr::pivot_longer(plot_data_rf, cols = c(Actual, Predicted), names_to = "Type", values_to = "Price")
plot_data_long_lasso <- tidyr::pivot_longer(plot_data_lasso, cols = c(Actual, Predicted), names_to = "Type", values_to = "Price")

connecting_lines_data_rf <- plot_data_long_rf %>%
  group_by(Index) %>%
  filter(n_distinct(Type) == 2)

connecting_lines_data_lasso <- plot_data_long_lasso %>%
  group_by(Index) %>%
  filter(n_distinct(Type) == 2)

plot_rf <- ggplot(connecting_lines_data_rf, aes(x = Index, y = Price, color = Type, group = Index)) +
  geom_point() +
  geom_line() +
  labs(x = "Index", y = "Price", title = "Actual vs. Predicted Price - Test - Random Forest")

plot_lasso <- ggplot(connecting_lines_data_lasso, aes(x = Index, y = Price, color = Type, group = Index)) +
  geom_point() +
  geom_line() +
  labs(x = "Index", y = "Price", title = "Actual vs. Predicted Price - Test - Lasso")

grid.arrange(plot_rf, plot_lasso, nrow = 2)

# EoF
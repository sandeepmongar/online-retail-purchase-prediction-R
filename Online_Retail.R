# Install and load required libraries
install.packages(c("caTools", "ROCR", "rpart", "caret", "randomForest", "ggplot2", "glmnet", "rpart.plot", "plotly","outliers","pROC"))
library(caTools)
library(ROCR)
library(rpart)
library(caret)
library(randomForest)
library(ggplot2)
library(glmnet)
library(rpart.plot)
library(plotly)
library(outliers)
library(pROC)

# Step 1: Load the dataset
df <- read.csv("online_retail_II.csv")

# Step 2: Explore the dataset
head(df)
summary(df)

# Step 3: Check for missing values
sum(is.na(df))
str(df)

# Check basic statistics
summary(df$Price)

# Identify outliers
outliers <- boxplot(df$Price, plot = FALSE)$out

# Display the identified outliers
cat("Identified outliers: ", outliers, "\n")

# We can remove outliers (if needed)
# df_no_outliers <- df[!(df$Price %in% outliers), ]

# Step 4: Convert 'InvoiceDate' to a datetime object
df$InvoiceDate <- as.POSIXct(df$InvoiceDate, format="%m/%d/%Y %H:%M")

# Step 5: Split the data into training and testing sets
set.seed(123)
split <- createDataPartition(df$Quantity, p = 0.7, list = FALSE)
train_data <- df[split, ]
test_data <- df[-split, ]

# Step 6: Convert 'Quantity' to a binary variable
train_data$Quantity <- ifelse(train_data$Quantity > 0, 1, 0)
test_data$Quantity <- ifelse(test_data$Quantity > 0, 1, 0)

# Step 7: Data Mining Techniques
# Convert 'Quantity' to a factor with two levels
train_data$Quantity <- factor(train_data$Quantity, levels = c(0, 1))
test_data$Quantity <- factor(test_data$Quantity, levels = c(0, 1))

# Convert 'Country' to a factor
train_data$Country <- as.factor(train_data$Country)
test_data$Country <- as.factor(test_data$Country)

# Ridge Regression with Cross-Validation and Hyperparameter Tuning
set.seed(123)
ridge_model_cv <- train(
  x = model.matrix(Quantity ~ Price + Country, data = train_data)[, -1],
  y = as.numeric(train_data$Quantity) - 1,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100))
)

# Print the best parameters
print(ridge_model_cv)

# Check the correlation between 'Price' and 'Country'
correlation <- cor(train_data$Price, as.numeric(train_data$Country))

# Print the correlation coefficient
print(correlation)

# Create design matrix for the test set
test_design_matrix <- model.matrix(as.formula(paste("Quantity ~", paste(c("Price", "Country"), collapse = " + "))), data = test_data)

# Make predictions on the test set
ridge_cv_predictions <- predict(ridge_model_cv, newdata = test_design_matrix[, -1], s = "lambda.min")
ridge_cv_binary_predictions <- ifelse(ridge_cv_predictions > 0.5, 1, 0)

# Convert probability predictions to binary predictions
ridge_cv_binary_predictions <- ifelse(ridge_cv_predictions > 0.5, 1, 0)

# Create a confusion matrix for ridge regression with cross-validation
ridge_cv_conf_matrix <- table(ridge_cv_binary_predictions, as.numeric(test_data$Quantity) - 1)
print("Ridge Regression Confusion Matrix:")
print(ridge_cv_conf_matrix)

# Calculate accuracy for ridge regression with cross-validation
ridge_cv_accuracy <- sum(diag(ridge_cv_conf_matrix)) / sum(ridge_cv_conf_matrix)
cat("Ridge Regression (CV) Accuracy:", ridge_cv_accuracy, "\n")

# Precision, Recall, and F1 Score for Ridge Regression
ridge_precision <- ridge_cv_conf_matrix[2, 2] / sum(ridge_cv_conf_matrix[, 2])
ridge_recall <- ridge_cv_conf_matrix[2, 2] / sum(ridge_cv_conf_matrix[2, ])
ridge_f1 <- 2 * (ridge_precision * ridge_recall) / (ridge_precision + ridge_recall)

cat("Ridge Regression Precision:", ridge_precision, "\n")
cat("Ridge Regression Recall:", ridge_recall, "\n")
cat("Ridge Regression F1 Score:", ridge_f1, "\n")

# Decision Tree with Cross-Validation and Hyperparameter Tuning
set.seed(123)
tree_model_cv <- train(
  x = train_data[, c("Price", "Country")],
  y = train_data$Quantity,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(cp = seq(0.001, 0.1, length = 10))
)

# Print the best parameters
print(tree_model_cv)

# Make predictions on the test set
tree_cv_predictions <- predict(tree_model_cv, newdata = test_data)

# Create a confusion matrix for decision tree with cross-validation
tree_cv_conf_matrix <- confusionMatrix(tree_cv_predictions, test_data$Quantity)
tree_cv_accuracy <- tree_cv_conf_matrix$overall["Accuracy"]
print(tree_cv_conf_matrix)
cat("Decision Tree (CV) Accuracy:", tree_cv_accuracy, "\n")

# Visualize decision tree
rpart.plot(tree_model$finalModel, type = 2, extra = 101, under = TRUE, main = "Decision Tree Model")

# Precision, Recall, and F1 Score for Decision Tree
tree_precision <- tree_cv_conf_matrix$byClass["Pos Pred Value"]
tree_recall <- tree_cv_conf_matrix$byClass["Sensitivity"]
tree_f1 <- tree_cv_conf_matrix$byClass["F1"]

cat("Decision Tree Precision:", tree_precision, "\n")
cat("Decision Tree Recall:", tree_recall, "\n")
cat("Decision Tree F1 Score:", tree_f1, "\n")

library(ggplot2)

# Histogram
histogram_plot <- ggplot(df, aes(x = Price)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  labs(title = "Histogram of Price", x = "Price", y = "Frequency")

# Convert to Plotly
histogram_plotly <- ggplotly(histogram_plot)

# Linear Plot with Price and Country
linear_plot <- ggplot(df, aes(x = Country, y = Price, color = Country)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Linear Plot of Price by Country", x = "Country", y = "Price") +
  theme_minimal()

# Convert Linear Plot to Plotly
linear_plotly <- ggplotly(linear_plot)


# Display the Plotly visualizations
print(histogram_plotly)
print(linear_plotly)


library(broom)

# Convert Ridge Regression coefficients to a tidy data frame
ridge_coef_df <- tidy(ridge_model_cv$finalModel)

# Visualize Ridge Regression Coefficients with Plotly
ridge_coef_plot <- ggplot(ridge_coef_df, aes(x = term, y = estimate)) +
  geom_col(fill = "skyblue") +
  labs(title = "Ridge Regression Model Coefficients", x = "Coefficient", y = "Value") +
  theme_minimal()

# Convert ggplot to Plotly
ridge_coef_plotly <- ggplotly(ridge_coef_plot)

# Display the Ridge Regression Coefficients plot
print(ridge_coef_plotly)

# Visualize Predictions with Plotly
ridge_prediction_plot <- ggplot() +
  geom_point(aes(x = seq_along(ridge_cv_binary_predictions), y = ridge_cv_binary_predictions), color = "blue") +
  labs(title = "Ridge Regression Predictions", x = "Observation", y = "Prediction") +
  theme_minimal()

# Convert ggplot to Plotly
ridge_prediction_plotly <- ggplotly(ridge_prediction_plot)

# Display the Ridge Regression Predictions plot
print(ridge_prediction_plotly)

# Visualize Decision Tree Predictions
tree_prediction_plot <- ggplot() +
  geom_point(aes(x = seq_along(tree_cv_predictions), y = tree_cv_predictions), color = "green") +
  labs(title = "Decision Tree Predictions", x = "Observation", y = "Prediction") +
  theme_minimal()

# Convert Decision Tree Predictions to Plotly
tree_prediction_plotly <- ggplotly(tree_prediction_plot)


# Visualize Model Metrics
metrics_data <- data.frame(
  Model = c("Ridge Regression", "Decision Tree"),
  Accuracy = c(ridge_cv_accuracy, tree_cv_accuracy),
  Precision = c(ridge_precision, tree_precision),
  Recall = c(ridge_recall, tree_recall),
  F1_Score = c(ridge_f1, tree_f1)
)

metrics_plot <- ggplot(metrics_data, aes(x = Model)) +
  geom_bar(aes(y = Accuracy, fill = "Accuracy"), position = "dodge", stat = "identity") +
  geom_bar(aes(y = Precision, fill = "Precision"), position = "dodge", stat = "identity") +
  geom_bar(aes(y = Recall, fill = "Recall"), position = "dodge", stat = "identity") +
  geom_bar(aes(y = F1_Score, fill = "F1 Score"), position = "dodge", stat = "identity") +
  labs(title = "Model Evaluation Metrics Comparison", x = "Model", y = "Metric Value") +
  theme_minimal() +
  scale_fill_manual(values = c("Accuracy" = "blue", "Precision" = "green", "Recall" = "red", "F1 Score" = "purple"))

# Convert Metrics Comparison to Plotly
metrics_plotly <- ggplotly(metrics_plot)

# Display the Metrics Comparison Plot
print(metrics_plotly)

# Display Plots
print(histogram_plotly)
print(boxplot_plotly)
print(ridge_coef_plotly)
print(tree_plot)
print(ridge_prediction_plotly)
print(tree_prediction_plotly)
print(accuracy_plotly)

# ROC Curve for Ridge Regression
roc_ridge <- roc(test_data$Quantity_numeric, ridge_cv_predictions)

# ROC Curve for Decision Tree
roc_tree <- roc(test_data$Quantity_numeric, as.numeric(tree_cv_predictions))

# Combine ROC curves in a single plot
roc_plot <- ggroc(list(ridge = roc_ridge, tree = roc_tree)) +
  theme_minimal() +
  labs(title = "ROC Curve Comparison", color = "Model")

print(roc_plot)





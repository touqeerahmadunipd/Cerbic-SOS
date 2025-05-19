# Clean workspace
rm(list = ls()); gc()
# Set working directory and load functions
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")

# Load data
###################################
##  phoneme data   ###############
#################################
set.seed(123)
data <- read.csv("phoneme.csv")
colnames(data)[which(names(data) == "Class")] <- "y"

# Split into train/test
library(caTools)
split <- sample.split(data$y, SplitRatio = 0.7)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# # OPTIONAL: Normalize (recommended for KDE, SMOTE)
# normalize_data <- function(train, test, target_col) {
#   features <- setdiff(names(train), target_col)
#   train_means <- apply(train[, features], 2, mean)
#   train_sds   <- apply(train[, features], 2, sd)
#   train_scaled <- scale(train[, features], center = train_means, scale = train_sds)
#   test_scaled  <- scale(test[, features], center = train_means, scale = train_sds)
#   train <- cbind(as.data.frame(train_scaled), y = train$y)
#   test  <- cbind(as.data.frame(test_scaled),  y = test$y)
#   list(train = train, test = test)
# }
# 
# scaled_data <- normalize_data(train_data, test_data, "y")
# train_data <- scaled_data$train
# test_data  <- scaled_data$test


# Balance test set

test_data_balanced <- ovun.sample(y~., data=test_data, p=0.5, seed=1, method="under")$data
# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)




# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_phoneme <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)

names(AM_rf_phoneme) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")


print(AM_rf_phoneme)












#########################################
##    House_16H data from openML   ####
######################################
# Load libraries
library(mlr3)
library(mlr3oml)

# Load the dataset from OpenML
dataset_id <- 821  # house_16H dataset ID on OpenML
house_16H <- OMLData$new(dataset_id)
data<-data.frame(house_16H$data)
colnames(data)[ncol(data)] <- "y"
# Extract the data as a data frame
table(data$y)
str(data$y)
# Convert factor levels to 0 and 1
data$y <- as.numeric(data$y == "N")  # "P" -> 1, "N" -> 0
# Check if the conversion worked
table(data$y)
# View summary of dataset



# Split the data (e.g., 70% training, 30% testing)
split <- sample.split(data$y, SplitRatio = 0.7)  # Assuming "Class" is the target variable

# Create training and testing sets
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Balance test set

test_data_balanced <- ovun.sample(y~., data=test_data, p=0.5, seed=1, method="under")$data
# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")


# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_house_16H <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)

names(AM_rf_house_16H) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_house_16H)






















#######################################
###     California data from OpenMl ###-----------------------------------------------------------------------------------
#######################################


# Load libraries
library(mlr3)
library(mlr3oml)

# Load the dataset from OpenML
dataset_id <- 45025  # California  data dataset ID on OpenML
california <- OMLData$new(dataset_id)
data<-data.frame(california$data)
colnames(data)[ncol(data)] <- "y"
# Extract the data as a data frame
table(data$y)
table(data$y)

# Split the data (e.g., 70% training, 30% testing)
split <- sample.split(data$y, SplitRatio = 0.7)  # Assuming "Class" is the target variable

# Create training and testing sets
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)


# Count the number of majority class (y == 0) and minority class (y == 1)
num_majority <- sum(train_data$y == 0)
num_minority <- sum(train_data$y == 1)

# Function to create imbalanced datasets
create_imbalance <- function(data, minority_ratio) {
  num_minority_new <- round(num_majority * minority_ratio)
  
  # Subset majority class (y == 0)
  majority_class <- subset(data, y == 0)
  
  # Randomly sample from minority class (y == 1)
  minority_class <- subset(data, y == 1)
  minority_class_sample <- minority_class[sample(nrow(minority_class), num_minority_new), ]
  
  # Combine and shuffle data
  imbalanced_data <- rbind(majority_class, minority_class_sample)
  imbalanced_data <- imbalanced_data[sample(nrow(imbalanced_data)), ]  # Shuffle
  
  return(imbalanced_data)
}


# Create datasets with different minority class ratios
train_data_20 <- create_imbalance(train_data, 0.20)
train_data_10 <- create_imbalance(train_data, 0.10)
train_data_05  <- create_imbalance(train_data, 0.05)

#
train_data<-train_data_20
test_data_balanced<- test_data


# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_california_0.20 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_california_0.20) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_california_0.20)











#10% imblancedness
train_data<-train_data_10

# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_california_0.10 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_california_0.10) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_california_0.10)







#5% imbalancedness
train_data<-train_data_05
# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_california_0.05 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_california_0.05) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_california_0.05)














#########################################
###  MagicTEl  from OpenML  ###############---------------------------------------
#########################################


# Load libraries
# Load the dataset from OpenML
dataset_id <- 43971  # MagicTel dataset ID on OpenML
MagicTel <- OMLData$new(dataset_id)
data<-data.frame(MagicTel$data)
colnames(data)[ncol(data)] <- "y"
# Extract the data as a data frame
table(data$y)
str(data$y)
# Convert factor levels to 0 and 1
data$y <- as.numeric(data$y == "h")  # "P" -> 1, "N" -> 0
# Check if the conversion worked
table(data$y)





# Split the data (e.g., 70% training, 30% testing)
split <- sample.split(data$y, SplitRatio = 0.7)  # Assuming "Class" is the target variable

# Create training and testing sets
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Check dimensions
dim(train_data)
dim(test_data)
table(train_data$y)
table(test_data$y)
##Balane the test set
# test_data_balanced <- ovun.sample(y~., data=test_data, 
#                                   p=0.5, seed=1, 
#                                   method="under")$data
# 
# head(test_data_balanced)
# table(test_data_balanced$y)

# Create datasets with different minority class ratios
train_data_20 <- create_imbalance(train_data, 0.20)
train_data_10 <- create_imbalance(train_data, 0.10)
train_data_05  <- create_imbalance(train_data, 0.05)

# Check class distributions
table(train_data_20$y)
table(train_data_10$y)
table(train_data_05$y)

##-----------------------------------------------------------
train_data<-train_data_20
test_data_balanced<- test_data


smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_MagicTel_0.20 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_MagicTel_0.20) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_MagicTel_0.20)








##10% imbalancedness-------------------------------------------------------------

train_data<-train_data_10

smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_MagicTel_0.10 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_MagicTel_0.10) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_MagicTel_0.10)







##5%imblancedness------------------------------------------------------------


train_data<-train_data_05


smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_MagicTel_0.05 <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_MagicTel_0.05) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_MagicTel_0.05)











#####################################
###### Abalone    ###################
########################################

#set.seed(123)
data<- read.csv("abalone.csv")
data$Class=ifelse(data$Rings>12, 1,0)
head(data)
data <- data[, !(names(data) %in% c("Sex", "Rings"))]
# Set seed for reproducibility
set.seed(123)

# Split the data (e.g., 70% training, 30% testing)
split <- sample.split(data$Class, SplitRatio = 0.7)  # Assuming "Class" is the target variable

# Create training and testing sets
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Check dimensions
dim(train_data)
dim(test_data)
colnames(train_data)[ncol(train_data)] <- "y"
colnames(test_data)[ncol(test_data)] <- "y"
table(train_data$y)
table(test_data$y)



# Balance test set

test_data_balanced <- ovun.sample(y~., data=test_data, p=0.5, seed=1, method="under")$data
head(test_data_balanced)
table(test_data_balanced$y)
# Balance training data
# SMOTE k = 5
smote_5 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(smote_5$x_new, y = as.factor(smote_5$y_new))

# SMOTE adaptive k
n1 <- sum(train_data$y == 1)
d <- ncol(train_data) - 1
k1 <- floor(n1^(4/(d+4)))
smote_k1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(smote_k1$x_new, y = as.factor(smote_k1$y_new))

# KDE-balanced training sets
train_data_kde  <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde$y  <- as.factor(train_data_kde$y)
train_data_kde1$y <- as.factor(train_data_kde1$y)

# Convert original train labels to factor
train_data$y <- as.factor(train_data$y)

# Leave test data unbalanced (realistic evaluation)
test_data$y <- as.factor(test_data$y)

# Train Random Forests
rf_smote     <- randomForest(y ~ ., data = train_data_smote)
rf_smote1    <- randomForest(y ~ ., data = train_data_smote1)
rf_kde       <- randomForest(y ~ ., data = train_data_kde)
rf_kde1      <- randomForest(y ~ ., data = train_data_kde1)

# Predict
rf_smote_pred  <- predict(rf_smote,  test_data_balanced[,-ncol(test_data_balanced)])
rf_smote1_pred <- predict(rf_smote1, test_data_balanced[,-ncol(test_data_balanced)])
rf_kde_pred    <- predict(rf_kde,    test_data_balanced[,-ncol(test_data_balanced)])
rf_kde1_pred   <- predict(rf_kde1,   test_data_balanced[,-ncol(test_data_balanced)])

# Prepare for CV
train_data1 <- train_data
train_data1$y <- factor(ifelse(train_data1$y == 0, "class0", "class1"))

# Define SMOTE and KDE sampling wrappers
smote_test <- list(
  name = "SMOTE with more neighbors!",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = 5)
    list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
  },
  first = TRUE
)


smote_test_k1 <- list(
  name = "SMOTE with more NN",
  func = function(x, y) {
    dat <- if (is.data.frame(x)) x else as.data.frame(x)
    dat$y <- y
    res <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], y = dat$y, k = k1)
    list(x = res$x_new, y = as.factor(res$y_new))
  },
  first = TRUE
)

kde_test_scott <- list(
  name = "KDE scott",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)

kde_test_scott_else <- list(
  name = "KDE scott_else",
  func = function(x, y) {
    dat <- balanced_data_kde1(x, y, rule = "scott_else")
    list(x = dat[,-ncol(dat)], y = as.factor(dat$y))
  },
  first = TRUE
)


# Cross-validation
ctrl_smote_5 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test)
ctrl_smote_k1 <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = smote_test_k1)
ctrl_smote_kde <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott)
ctrl_smote_kde_else <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = kde_test_scott_else)

tuneGrid <- expand.grid(mtry = seq(2, ncol(train_data) - 1, by = 1))
rf_smote_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_5, tuneGrid = tuneGrid,  metric = "ROC")
rf_smote_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_k1, tuneGrid = tuneGrid, metric = "ROC")
rf_kde_cv <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde,  tuneGrid = tuneGrid,metric = "ROC")
rf_kde_cv1 <- train(y ~ ., data = train_data1, method = "rf", trControl = ctrl_smote_kde_else, tuneGrid = tuneGrid, metric = "ROC")

# Final RFs using best mtry
rf_smote_cv_final  <- randomForest(y ~ ., data = train_data_smote,  mtry = rf_smote_cv$bestTune$mtry)
rf_smote_cv1_final <- randomForest(y ~ ., data = train_data_smote1, mtry = rf_smote_cv1$bestTune$mtry)
rf_kde_cv_final    <- randomForest(y ~ ., data = train_data_kde,    mtry = rf_kde_cv$bestTune$mtry)
rf_kde_cv1_final   <- randomForest(y ~ ., data = train_data_kde1,   mtry = rf_kde_cv1$bestTune$mtry)

# Predict
rf_pred_smote_cv   <- predict(rf_smote_cv_final,  test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_smote_cv1  <- predict(rf_smote_cv1_final, test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv     <- predict(rf_kde_cv_final,    test_data_balanced[,-ncol(test_data_balanced)])
rf_pred_kde_cv1    <- predict(rf_kde_cv1_final,   test_data_balanced[,-ncol(test_data_balanced)])

# Confusion Matrices
conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv  <- confusionMatrix(rf_pred_smote_cv,  as.factor(test_data_balanced$y), mode = "everything")
conf_rf_smote_cv1 <- confusionMatrix(rf_pred_smote_cv1, as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv    <- confusionMatrix(rf_pred_kde_cv,    as.factor(test_data_balanced$y), mode = "everything")
conf_rf_kde_cv1   <- confusionMatrix(rf_pred_kde_cv1,   as.factor(test_data_balanced$y), mode = "everything")

# Extract (1 - Balanced Accuracy)
AM_rf_abalone <- c(
  1 - conf_rf_smote$byClass["Balanced Accuracy"],
  1 - conf_rf_smote1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde$byClass["Balanced Accuracy"],
  1 - conf_rf_kde1$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_smote_cv1$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv$byClass["Balanced Accuracy"],
  1 - conf_rf_kde_cv1$byClass["Balanced Accuracy"]
)
names(AM_rf_abalone) <- c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

print(AM_rf_abalone)









# Load required library
library(ggplot2)

# Define methods (each dataset has these 5 methods)
methods <- rep( c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV"), 9)

# Define dataset names (each appears 5 times)
# datasets <- rep(c("Phoneme", "Abalone", "House_16H", "California_0.01", "California_0.1", 
#                   "California_0.2", "MagicTel_0.01", "MagicTel_0.1", "MagicTel_0.2"), each = 10)
datasets <- rep(c("Phoneme","Abalone", "House_16H", "California(5%)", "California(10%)", 
                  "California(30%)", "MagicTel(5%)", "MagicTel(10%)", "MagicTel(30%)"), each = 8)


AM_values <- c(
  AM_rf_phoneme[1:8],
  AM_rf_abalone[1:8],
  AM_rf_house_16H[1:8],
  AM_rf_california_0.05[1:8],
  AM_rf_california_0.10[1:8],
  AM_rf_california_0.20[1:8],
  AM_rf_MagicTel_0.05[1:8],
  AM_rf_MagicTel_0.10[1:8],
  AM_rf_MagicTel_0.20[1:8]
)
# Create a data frame
data <- data.frame(Methods = methods, Dataset = datasets, AM_Risk = AM_values)


##
library(ggplot2)
library(dplyr)

# Ensure Dataset and Methods are factors
data$Dataset <- factor(data$Dataset, 
                       levels = c("Abalone","California(30%)", "California(10%)", "California(5%)", 
                                  "MagicTel(30%)", "MagicTel(10%)", "MagicTel(5%)", 
                                  "Phoneme", "House_16H"))

desired_order <-  c("SMOTE(S)","SMOTE(L)", "KDE(L)", "KDE(S)", "SMOTE(S)-CV","SMOTE(L)-CV", "KDE(L)-CV",  "KDE(S)-CV")

# Reorder methods by desired order within each dataset
data_ordered <- data %>%
  mutate(Methods = factor(Methods, levels = desired_order)) %>%
  arrange(Dataset, Methods)

# View result
print(data_ordered)

data=data_ordered


save(data, file="AM-risk-realapp_RF.RData")
load("AM-risk-realapp_RF.RData")
# Flag minimum values per dataset
data_annotated <- data %>%
  group_by(Dataset) %>%
  mutate(is_min = AM_Risk == min(AM_Risk))


data_annotated$Methods <- factor(data_annotated$Methods,
                                 levels = c("KDE(SB)", "SMOTE(k=5)", "KDE", "SMOTE(k=v)", "KDE-CV(SB)", "SMOTE-CV(k=5)", "KDE-CV", "SMOTE-CV(k=v)"),
                                 labels = c("KDE(S)", "SMOTE(S)", "KDE(L)", "SMOTE(L)", "KDE(S)-CV", "SMOTE(S)-CV", "KDE(L)-CV", "SMOTE(L)-CV")
)

# Plot: Dataset on x-axis, bars for each method
p <- ggplot(data_annotated, aes(x = Dataset, y = AM_Risk, fill = Methods)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  # Add text annotation for minimum values
  # geom_text(data = filter(data_annotated, is_min),
  #           aes(label = round(AM_Risk, 3)),
  #           position = position_dodge(width = 0.9),
  #           vjust = -0.5,  # Position above the bars
  #           size = 3,
  #           color = "black") +
  # Add the title and axis labels
  labs(title = "AM Risk by Different Methods for Different Datasets Using RandomForest",
       x = "Datasets", y = "AM Risk") +
  # Apply minimal theme
  
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.3, "cm"),
    legend.spacing.x = unit(0.3, 'cm')
  ) +  # Remove the legend for colors
  scale_fill_brewer(palette = "Set3")  # Apply a color palette

# Display the plot
p

manual_colors <- c(
  # "Weighted"        = "#E63946",  # strong red
  "KDE(S)"         = "#457B9D",  # cool blue
  "SMOTE(S)"      = "#2A9D8F",  # teal green
  "KDE(L)"             = "#F4A261",  # warm orange
  "SMOTE(L)"      = "#A8DADC",  # pale cyan
  "KDE(S)-CV"      = "blue",  # soft gray-blue
  "SMOTE(S)-CV"   = "#F72585",  # hot pink
  "KDE(L)-CV"          = "#7209B7",  # deep violet
  "SMOTE(L)-CV"   = "#06D6A0"   # mint green
)

p <- ggplot(data_annotated, aes(x = Dataset, y = AM_Risk, fill = Methods)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  labs(title = "AM Risk when using RandomForest",
       x = "Datasets", y = "AM Risk") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.3, "cm"),
    legend.spacing.x = unit(0.3, 'cm')
  ) +
  scale_fill_manual(values = manual_colors)

p

ggsave("Real-data-plot_RF11.pdf", plot = p, width = 6, height = 6, dpi = 300)##AM_risk_standard_knn_with_balanced_test------------------------------





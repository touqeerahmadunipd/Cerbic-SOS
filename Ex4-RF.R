rm(list=ls());gc()
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")


##Test set----------------------------------------------
set.seed(123)  # For reproducibility

# Parameters
# table(df$y)
generate_clustered_data <- function(p, n_points = 1000) {
  n_clusters <- 4
  
  # Given probability for each cluster
  p1 <- c(p/2, (1 - p)/2, (1 - p)/2, p/2)
  
  # Normalize (defensive step)
  p1 <- p1 / sum(p1)
  
  # Compute cluster sizes
  cluster_sizes <- round(p1 * n_points)
  
  # Fix rounding issues if total != n_points
  cluster_sizes[length(cluster_sizes)] <- n_points - sum(cluster_sizes[-length(cluster_sizes)])
  n_samples <- sum(cluster_sizes)
  
  # Initialize matrices
  X <- matrix(0, nrow = n_samples, ncol = 2)
  y <- numeric(n_samples)
  
  # Cluster centers
  centers <- matrix(c(0, 0, 10, 0, 0, 10, 10, 10), ncol = 2, byrow = TRUE)
  
  # Assign points to clusters with varying sizes
  start_idx <- 1
  for (i in 1:n_clusters) {
    end_idx <- start_idx + cluster_sizes[i] - 1
    X[start_idx:end_idx, ] <- sweep(matrix(rnorm(cluster_sizes[i] * 2, mean = 0, sd = 3), 
                                           ncol = 2), 2, centers[i, ], "+")
    if (i == 1 || i == 4) {
      y[start_idx:end_idx] <- 0
    } else {
      y[start_idx:end_idx] <- 1
    }
    start_idx <- end_idx + 1
  }
  
  # Prepare dataframe to return
  data <- data.frame(X1 = X[,1], X2 = X[,2], y = as.numeric(as.character(y)))
  
  return(data)
}



# Set number of iterations
R <- 50

# Function to initialize and name matrices
init_matrices <- function(R, cols, colnames_list) {
  mats <- list(
    alpha_01 = matrix(NA, R, cols),
    alpha_02 = matrix(NA, R, cols),  #01 mean -1
    alpha_03 = matrix(NA, R, cols),
    alpha_04 = matrix(NA, R, cols),
    alpha_05 = matrix(NA, R, cols),
    alpha_06 = matrix(NA, R, cols)
  )
  for (mat in names(mats)) {
    colnames(mats[[mat]]) <- colnames_list
  }
  mats
}
# Column names for AM metrics matrices
AM_colnames <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
#AM risk
AM_risk_rf <- init_matrices(R, 4, AM_colnames)




##Test set-----------------------------
n.size_test <- 10000
p=0.60
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_01[r, ] <- AM_rf
  
}
close(pb)





#Second setting---------------------------------------------------------


##Test set-----------------------------
n.size_test <- 10000
p=0.80
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_02[r, ] <- AM_rf
  
}
close(pb)












#Third setting---------------------------------------------------------


##Test set-----------------------------
n.size_test <- 10000
p=0.85
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_03[r, ] <- AM_rf
  
}
close(pb)

















#Forth setting---------------------------------------------------------


##Test set-----------------------------
n.size_test <- 10000
p=0.90
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_04[r, ] <- AM_rf
  
}
close(pb)








#fifth setting---------------------------------------------------------


##Test set-----------------------------
n.size_test <- 10000
p=0.92
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_05[r, ] <- AM_rf
  
}
close(pb)






#sixth setting---------------------------------------------------------


##Test set-----------------------------
n.size_test <- 10000
p=0.95
test_data_imbalance <- generate_clustered_data(p=p, n_points = n.size_test)


test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)


##Training----------------------------------------------------


# Set parameters
n.size <- 1000

# Initialize progress bar
pb <- txtProgressBar(min = 0, max = R, style = 3)

# Simulation loop
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Parameters
  
  data <- generate_clustered_data(p=p, n_points = n.size)
  train_data <- data
  
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
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
    train_data_kde  <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde$y  <- as.factor(train_data_kde$y)
    train_data_kde1$y <- as.factor(train_data_kde1$y)
  }
  # Convert original train labels to factor
  train_data$y <- as.factor(train_data$y)
  
  # Leave test data unbalanced (realistic evaluation)
  
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
  
  #balanced rf with AM risk------------------------------------------------------------------------------
  # Confusion Matrices
  conf_rf_smote     <- confusionMatrix(rf_smote_pred,     as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_smote1    <- confusionMatrix(rf_smote1_pred,    as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde       <- confusionMatrix(rf_kde_pred,       as.factor(test_data_balanced$y), mode = "everything")
  conf_rf_kde1      <- confusionMatrix(rf_kde1_pred,      as.factor(test_data_balanced$y), mode = "everything")
  
  # Extract (1 - Balanced Accuracy)
  AM_rf<- c(
    1 - conf_rf_smote$byClass["Balanced Accuracy"],
    1 - conf_rf_smote1$byClass["Balanced Accuracy"],
    1 - conf_rf_kde$byClass["Balanced Accuracy"],
    1 - conf_rf_kde1$byClass["Balanced Accuracy"]
  )
  
  names(AM_rf) <- c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_rf$alpha_06[r, ] <- AM_rf
  
}
close(pb)





# Mean AM risk rf------------------------------------
AM_risk_rf_alpha_01<- round(colMeans(na.omit(AM_risk_rf$alpha_01)), 4)
AM_risk_rf_alpha_02<- round(colMeans(na.omit(AM_risk_rf$alpha_02)), 4)
AM_risk_rf_alpha_03<- round(colMeans(na.omit(AM_risk_rf$alpha_03)), 4)
AM_risk_rf_alpha_04<- round(colMeans(na.omit(AM_risk_rf$alpha_04)), 4)
AM_risk_rf_alpha_05<- round(colMeans(na.omit(AM_risk_rf$alpha_05)), 4)
AM_risk_rf_alpha_06<- round(colMeans(na.omit(AM_risk_rf$alpha_06)), 4)






#ggsave("ECSD-exp-gpd-kde-nn.pdf", plot = p, width = 5, height = 5, dpi = 300)##AM_risk_standard_rf_with_balanced_test------------------------------

library(ggplot2)

# Define the methods and labels
methods <- c("0.60","0.80","0.85", "0.90", "0.92", "0.95")
labels <-  c("SMOTE(S)", "SMOTE(L)", "KDE(L)", "KDE(S)")

##AM_risk_standard_rf_with_imbalanced_test----------

dfs <- lapply(1:4, function(i) {
  data.frame(
    d = rep(labels[i], 6),
    method = methods,
    x = c(
      AM_risk_rf_alpha_01[i],
      AM_risk_rf_alpha_02[i],
      AM_risk_rf_alpha_03[i],
      AM_risk_rf_alpha_04[i],
      AM_risk_rf_alpha_05[i],
      AM_risk_rf_alpha_06[i]
    )
  )
})


# Combine all data frames
AM_risk_rf <- do.call(rbind, dfs)

# Convert "method" to ordered factor with custom levels
AM_risk_rf$method <- factor(
  AM_risk_rf$method,
  levels = methods,
  ordered = TRUE
)

# Reorder the levels of "d" variable
AM_risk_rf$d <- factor(
  AM_risk_rf$d,
  levels = labels
)


AM_risk_rf$t <-  "Example 4"
# Assign shapes (e.g., cross = 4, dot = 16, triangle = 17, etc.)
manual_shapes <- c(
  #"BBC" = 4,        # cross
  #"BBC-CV" = 16,    # dot
  "SMOTE(S)" = 16,      # cross
  "SMOTE(L)" = 4,     # dot
  #"SMOTE(S)-CV" = 16,
  #"SMOTE(L)-CV" = 4,
  "KDE(L)" = 4,
  "KDE(S)" = 16
  #"KDE(L)-CV" = 4,
  #"KDE(S)-CV" = 16
)

# Define custom colors (same for related methods)
manual_colors <- c(
  #"BBC" = "#1b9e77",       # greenish
  #"BBC-CV" = "#1b9e77",    
  "SMOTE(S)" = "#d95f02",     # orange
  "SMOTE(L)" = "#d95f02",     
  #"SMOTE(S)-CV" = "#7570b3",  # purple
  #"SMOTE(L)-CV" = "#7570b3",  
  "KDE(L)" = "#e7298a",            # pink
  "KDE(S)" = "#e7298a"
  #"KDE(L)-CV" = "#66a61e",         # green
  #"KDE(S)-CV" = "#66a61e"
)

# Plot

p <- ggplot(AM_risk_rf, aes(x = method, y = x, group = d, color = d, shape = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 2) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5) +
  labs(x = "Quantiles", y = "AM Risk", title = "") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    legend.key.size = unit(0.5, "lines"),
    legend.title = element_text(),  # removed face = "bold"
    legend.text = element_text(size = 8),
    strip.text = element_text(size = 12)  # removed face = "bold"
  ) +
  scale_color_manual(
    name = "Methods", 
    values = manual_colors,
    guide = guide_legend(nrow = 1)
  ) +
  scale_shape_manual(
    name = "Methods", 
    values = manual_shapes,
    guide = guide_legend(nrow = 1)
  ) +
  facet_wrap(~ t, nrow = 1, scales = "free_x")


p



ggsave("AM-Exm4-RF.pdf", plot = p, width = 6, height = 4, dpi = 300)##AM_risk_standard_rf_with_balanced_test------------------------------


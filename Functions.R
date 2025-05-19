#rm(list=ls());gc()

library(RANN)
library(glmnet)
library(dplyr)
library(pbapply)
library(class)
library(mev)
library(caret)
library(SMOTEWB)
library(MASS)
library(expm)
library(FNN)
library(smotefamily)
library(randomForest)
library(kernelboot)
library(ROSE)
library(caTools)
library(ggplot2)
library(gplm)

set.seed(123)
#remove.packages("FNN", lib = "/home/atouqeer/R/x86_64-pc-linux-gnu-library/4.4")

compute_knn_radius <- function(x, train, k) {
  x <- as.numeric(x)
  train <- as.matrix(train)
  distances <- sqrt(rowSums((train - matrix(x, nrow = nrow(train), ncol = length(x), byrow = TRUE))^2))
  sorted_distances <- sort(distances)
  radius <- sorted_distances[k]
  return(radius)
}



# Function to find indices of points within the k-NN radius
find_knn_indices <- function(x, train, radius) {
  x <- as.numeric(x)
  train <- as.matrix(train)
  distances <- sqrt(rowSums((train - matrix(x, nrow = nrow(train), ncol = length(x), byrow = TRUE))^2))
  indices <- which(distances <= radius)
  return(indices)
}





# Main balanced k-NN function
balanced_knn <- function(train, trainlabels, test, k, p_hat=NULL) {
  n_test <- nrow(test)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    x <- as.numeric(test[i, , drop = FALSE])
    # Compute the k-NN radius
    radius <- compute_knn_radius(x, train, k)
    # Find the k-NN indices
    knn_indices <- find_knn_indices(x, train, radius)
    # Calculate the estimate of the regression function
    eta_hat <- mean(trainlabels[knn_indices])
    # Apply the majority vote rule
    predictions[i] <- ifelse(eta_hat >= p_hat, 1, 0)
  }
  
  return(as.factor(predictions))
}

##Apply weights on labels
balanced_knn2 <- function(train, trainlabels, test, k, w = 0.5) {
  n_test <- nrow(test)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    x <- as.numeric(test[i, , drop = FALSE])
    
    # Compute the k-NN radius
    radius <- compute_knn_radius(x, train, k)
    
    # Find the k-NN indices
    knn_indices <- find_knn_indices(x, train, radius)
    
    # Get labels of neighbors
    neighbor_labels <- trainlabels[knn_indices]
    
    # Count number of neighbors in each class
    n1 <- sum(neighbor_labels == 1)
    n0 <- sum(neighbor_labels == 0)
    
    # Weighted decision rule
    predictions[i] <- ifelse(w * n1 >= (1 - w) * n0, 1, 0)
  }
  
  return(as.factor(predictions))
}


cross_validate_bknn <- function(train, trainlabels, k_values, m_folds=5) {
  n <- nrow(train)
  fold_size <- floor(n / m_folds)
  performance <- matrix(0, nrow = m_folds, ncol = length(k_values))
  
  # Create folds
  set.seed(123)  # Set seed for reproducibility
  folds <- split(sample(1:n), rep(1:m_folds, each = fold_size, length.out = n))
  
  for (i in 1:m_folds) {
    # Split data into training and validation sets
    validation_indices <- folds[[i]]
    train_indices <- setdiff(1:n, validation_indices)
    
    train_fold <- train[train_indices, ]
    trainlabels_fold <- trainlabels[train_indices]
    validation_fold <- train[validation_indices, ]
    validationlabels_fold <- trainlabels[validation_indices]
    
    data <- data.frame(validation_fold, y=validationlabels_fold)
    test_data_balanced_cv <- ovun.sample(y~., data=data, 
                                         p=0.5, seed=1, 
                                         method="under")$data
    
    validation_fold <- test_data_balanced_cv[,-ncol(test_data_balanced_cv)]
    validationlabels_fold <- test_data_balanced_cv$y
    
    p_hat <- mean(trainlabels_fold==1)
    
    for (j in 1:length(k_values)) {
      k <- k_values[j]
      # Make predictions on the validation set
      predictions <- balanced_knn(train_fold, trainlabels_fold, validation_fold, k=k, p_hat)
      # Compute accuracy (or any other performance metric)
      conf_mat <- confusionMatrix(predictions, as.factor(validationlabels_fold), mode = "everything")
      AM_risk <- 1 - conf_mat$byClass[11]
      performance[i, j] <- AM_risk
    }
  }
  
  # Compute average performance for each k
  avg_performance <- colMeans(performance)
  
  # Find the k with the highest average performance
  best_k <- k_values[which.max(avg_performance)]
  
  return(list(best_k = best_k, avg_performance = avg_performance))
}

#Weightes based cross validation
cross_validate_bknn2 <- function(train, trainlabels, k_values, m_folds=5) {
  n <- nrow(train)
  fold_size <- floor(n / m_folds)
  performance <- matrix(0, nrow = m_folds, ncol = length(k_values))
  
  # Create folds
  set.seed(123)  # Set seed for reproducibility
  folds <- split(sample(1:n), rep(1:m_folds, each = fold_size, length.out = n))
  
  for (i in 1:m_folds) {
    # Split data into training and validation sets
    validation_indices <- folds[[i]]
    train_indices <- setdiff(1:n, validation_indices)
    
    train_fold <- train[train_indices, ]
    trainlabels_fold <- trainlabels[train_indices]
    validation_fold <- train[validation_indices, ]
    validationlabels_fold <- trainlabels[validation_indices]
    
    data <- data.frame(validation_fold, y = validationlabels_fold)
    test_data_imbalanced_cv <- data  # Keep original fold
    
    validation_fold <- test_data_imbalanced_cv[, -ncol(test_data_imbalanced_cv)]
    validationlabels_fold <- test_data_imbalanced_cv$y
    
    # Ensure consistent levels
    validationlabels_fold <- factor(validationlabels_fold, levels = c("0", "1"))
    trainlabels_fold <- factor(trainlabels_fold, levels = c("0", "1"))
    
    p_hat <- mean(trainlabels_fold == "1")
    
    for (j in 1:length(k_values)) {
      k <- k_values[j]
      # Make predictions on the validation set
      predictions <- balanced_knn2(train_fold, trainlabels_fold, validation_fold, k = k, w = p_hat)
      predictions <- factor(predictions, levels = c("0", "1"))
      
      conf_mat <- confusionMatrix(predictions, validationlabels_fold, mode = "everything")
      
      if ("Accuracy" %in% names(conf_mat$overall)) {
        AM_risk <- 1 - conf_mat$overall["Accuracy"]
        performance[i, j] <- AM_risk
      } else {
        warning(paste("Skipping fold", i, "k =", k, "- Accuracy not found"))
      }
    }
  }
  
  # Compute average performance for each k
  avg_performance <- colMeans(performance, na.rm = TRUE)
  
  # Find the k with the highest average performance
  best_k <- k_values[which.max(avg_performance)]
  
  return(list(best_k = best_k, avg_performance = avg_performance))
}


##

# check if different objects are numeric
# for data.frames and matrix objects check the individual columns

numericColumns <- function(x) UseMethod("numericColumns")

numericColumns.default <- function(x) {
  is.numeric(x)
}

numericColumns.matrix <- function(x) {
  structure(rep(is.numeric(x), ncol(x)), names = colnames(x))
}

numericColumns.data.frame <- function(x) {
  unlist(lapply(x, is.numeric))
}

# check for square matrix

is.square <- function(x) {
  NCOL(x) == NROW(x)
}

# check for diagonal matrix

is.diag <- function(x) {
  dx <- diag(diag(x))
  is.square(x) && all(abs(dx - x) <= .Machine$double.eps)
}

# test for being the vector

is.simple.vector <- function(x) {
  is.atomic(x) && !is.recursive(x) && !is.array(x)
}

# test is all elements are zeros

is.allzeros <- function(x) {
  all(abs(x) <= .Machine$double.eps)
}

##
# a=0.5
bw.scott_else<- function (x, na.rm = FALSE)
{
  
  if (!(is.matrix(x) || is.data.frame(x)))
    stop("this method works only for matrix, or data.frame objects")
  if (!all(numericColumns(x)))
    stop("all columns need to be numeric")
  if (na.rm)
    x <- na.omit(x)
  m <- ncol(x)
  n <- nrow(x)
  S <- var(x)
  n^(-2/(m+4 )) * S
}


# 




balanced_data_kde <- function(x, y, rule=c("scott","scott_else")) {
  # Check if 'y' is a factor, otherwise convert it
  data<- data.frame(x, y)
  if (!is.factor(data$y)) {
    y <- as.factor(data$y)
  }
  
  
  # Count the number of instances in each class
  class_counts <- table(data$y)
  
  # Identify the minority and majority classes
  minority_class <- names(class_counts)[which.min(class_counts)]
  majority_class <- names(class_counts)[which.max(class_counts)]
  
  # Separate data into majority and minority classes
  X_majority <- data %>% filter(y == majority_class)
  X_minority <- data %>% filter(y == minority_class)
  
  # Convert the minority class data (without 'y') to a matrix
  #minority_data <- as.matrix(X_minority %>% select(-y))
  
  X_minority_X<-X_minority
  X_minority_X<- X_minority_X[-ncol(X_minority_X)]
  # Convert the minority class data (without 'y') to a matrix
  minority_data <- as.matrix(X_minority_X )
  
  
  # Number of synthetic samples needed to balance the dataset
  num_synthetic_samples <- nrow(X_majority) - nrow(X_minority)
  
  # 
  
  
  # Generate synthetic samples
  # synthetic_minority_samples <- generate_synthetic_samples(kde_minority, num_synthetic_samples)
  
  
  #synthetic_minority_samples <- rkde_scott(minority_data, num_synthetic_samples)
  if (rule=="scott"){
    synthetic_minority_samples <- kernelboot::rmvg(num_synthetic_samples, minority_data, bw =bw.scott(minority_data), weights = NULL, adjust = 1)
  } else if (rule=="scott_else") {
    synthetic_minority_samples <- kernelboot::rmvg(num_synthetic_samples, minority_data, bw =(1/10)*bw.scott(minority_data), weights = NULL, adjust = 1)
  }
  #synthetic_minority_samples <- rkde_first_nn1(minority_data, num_synthetic_samples)
  # Convert the synthetic samples to a data frame and add the minority class label
  synthetic_minority_df <- data.frame(synthetic_minority_samples)
  colnames(synthetic_minority_df) <- colnames(minority_data)# %>% select(-y))  # Use original covariate names
  synthetic_minority_df$y <- as.factor(minority_class)
  
  # Combine original majority, minority, and synthetic samples
  balanced_data <- rbind(X_majority, X_minority, synthetic_minority_df)
  colnames(balanced_data) <- colnames(data)
  
  # Return the balanced dataset
  return(balanced_data)
}




balanced_data_kde1 <- function(x, y, rule=c("scott","scott_else")) {
  # Check if 'y' is a factor, otherwise convert it
  data<- data.frame(x, y)
  if (!is.factor(data$y)) {
    y <- as.factor(data$y)
  }
  
  
  # Count the number of instances in each class
  class_counts <- table(data$y)
  
  # Identify the minority and majority classes
  minority_class <- names(class_counts)[which.min(class_counts)]
  majority_class <- names(class_counts)[which.max(class_counts)]
  
  # Separate data into majority and minority classes
  X_majority <- data %>% filter(y == majority_class)
  X_minority <- data %>% filter(y == minority_class)
  
  # Convert the minority class data (without 'y') to a matrix
  #minority_data <- as.matrix(X_minority %>% select(-y))
  
  X_minority_X<-X_minority
  X_minority_X<- X_minority_X[-ncol(X_minority_X)]
  # Convert the minority class data (without 'y') to a matrix
  minority_data <- as.matrix(X_minority_X )
  
  
  # Number of synthetic samples needed to balance the dataset
  num_synthetic_samples <- nrow(X_majority) - nrow(X_minority)
  
  # 
  
  
  # Generate synthetic samples
  # synthetic_minority_samples <- generate_synthetic_samples(kde_minority, num_synthetic_samples)
  
  
  #synthetic_minority_samples <- rkde_scott(minority_data, num_synthetic_samples)
  if (rule=="scott"){
    synthetic_minority_samples <- kernelboot::rmvg(num_synthetic_samples, minority_data, bw =bandwidth.scott(minority_data), weights = NULL, adjust = 1)
  } else if (rule=="scott_else") {
    synthetic_minority_samples <- kernelboot::rmvg(num_synthetic_samples, minority_data, bw =(1/10)*bandwidth.scott(minority_data), weights = NULL, adjust = 1)
  }
  #synthetic_minority_samples <- rkde_first_nn1(minority_data, num_synthetic_samples)
  # Convert the synthetic samples to a data frame and add the minority class label
  synthetic_minority_df <- data.frame(synthetic_minority_samples)
  colnames(synthetic_minority_df) <- colnames(minority_data)# %>% select(-y))  # Use original covariate names
  synthetic_minority_df$y <- as.factor(minority_class)
  
  # Combine original majority, minority, and synthetic samples
  balanced_data <- rbind(X_majority, X_minority, synthetic_minority_df)
  colnames(balanced_data) <- colnames(data)
  
  # Return the balanced dataset
  return(balanced_data)
}


#Multivariate Gaussian SMOTE

multivariate_gaussian_smote <- function(x, y, k = 5) {
  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (!is.factor(y)) {
    stop("y must be a factor")
  }
  
  var_names <- colnames(x)
  p <- ncol(x)
  class_names <- levels(y)
  x_syn_list <- list()
  
  for (label in class_names) {
    x_class <- x[y == label, , drop = FALSE]
    n_needed <- max(table(y)) - sum(y == label)
    
    for (i in 1:n_needed) {
      # Step 1: Select a random sample from the minority class
      x_sample <- x_class[sample(1:nrow(x_class), 1), , drop = FALSE]
      
      # Step 2: Compute k-NN mean and standard deviation
      NN_index <- FNN::get.knnx(data = x, query = x_sample, k = k + 1)$nn.index[, -1]
      x_neighbors <- x[NN_index, , drop = FALSE]
      
      mu_hat <- colMeans(x_neighbors)
      H_hat <- apply(x_neighbors, 2, sd)
      
      # Step 3: Generate a random Gaussian noise sample
      Z_star <- rnorm(p, mean = 0, sd = 1)
      
      # Step 4: Generate synthetic sample
      x_synthetic <- mu_hat + H_hat * Z_star
      
      x_syn_list <- append(x_syn_list, list(x_synthetic))
    }
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(rep(class_names, times = max(table(y)) - table(y)), levels = class_names)
  
  # Combine with original data
  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)
  colnames(x_new) <- var_names
  
  return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}



###Data Generator function for example 4

# Parameters
# table(df$y)
generate_clustered_data <- function(p, n_points) {
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


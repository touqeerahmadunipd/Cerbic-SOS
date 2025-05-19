rm(list=ls());gc()
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")


set.seed(123)
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
AM_colnames <- c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
AM_colnames_logistic <- c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
#AM risk
AM_risk_knn <- init_matrices(R, 10, AM_colnames)
AM_risk_logistic <- init_matrices(R, 5, AM_colnames_logistic)



#Fist setting

##Test set----------------

n.size_test <- 10000
p <-4

X_test <- matrix(rnorm(p* n.size_test), nrow = n.size_test)
colnames(X_test) <- paste0("x", 1:p)
X_test<- data.frame(X_test)
Xmat_test <- model.matrix( ~.,data=X_test)
beta1 <- c(0,1, rep(0, p-1))
alpha=-1
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-   c("Weighted", "Weighted_CV","SMOTE", "SMOTE1", "SMOTE-CV","SMOTE-CV1","KDE", "KDE1","KDE-CV","KDE-CV1")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_01[r, ] <- AM_knn
  AM_risk_logistic$alpha_01[r, ] <- AM_logistic
  
  
}
close(pb)





#Setting 2-----------------------------------------------------------------



alpha=-1.5
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  #knn_Weighted_standard_with_imbalanced_test<- class::knn(train = train_data[, -ncol(train_data)], test = test_data_balanced[,- ncol(test_data_balanced)] ,cl =train_data$y,k=k_range)
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-   c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_02[r, ] <- AM_knn
  AM_risk_logistic$alpha_02[r, ] <- AM_logistic
  
  
}
close(pb)





#setting 3-----------------------------------------------------------------



alpha=-2
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  #knn_Weighted_standard_with_imbalanced_test<- class::knn(train = train_data[, -ncol(train_data)], test = test_data_balanced[,- ncol(test_data_balanced)] ,cl =train_data$y,k=k_range)
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-  c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_03[r, ] <- AM_knn
  AM_risk_logistic$alpha_03[r, ] <- AM_logistic
  
  
}
close(pb)



#Setting 4----------------------------------------------------------------------------



alpha=-2.5
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-   c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_04[r, ] <- AM_knn
  AM_risk_logistic$alpha_04[r, ] <- AM_logistic
  
  
}
close(pb)



#Setting 5---------------------------------------------------------------------------



alpha=-3
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  #knn_Weighted_standard_with_imbalanced_test<- class::knn(train = train_data[, -ncol(train_data)], test = test_data_balanced[,- ncol(test_data_balanced)] ,cl =train_data$y,k=k_range)
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-  c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_05[r, ] <- AM_knn
  AM_risk_logistic$alpha_05[r, ] <- AM_logistic
  
  
}
close(pb)





#Setting 6--------------------------------------------------------------------------



alpha=-3.5
#Test------
prob_test<-rje::expit((Xmat_test %*% beta1)+alpha)
y <- rbinom(n.size_test, size = 1, prob = prob_test)
table(y)
t_data <- cbind(X_test,y = y)
test_data_imbalance<-t_data
table(test_data_imbalance$y)


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
  p=4
  # Generate predictors
  # Generate predictors
  X <- matrix(rnorm(p* n.size), nrow = n.size)
  colnames(X) <- paste0("x", 1:p)
  X<- data.frame(X)
  Xmat <- model.matrix( ~.,data=X)
  beta1 <- c(0,1, rep(0, p-1))
  # Thresholds
  
  prob<-rje::expit((Xmat %*% beta1)+alpha)
  y <- rbinom(n.size, size = 1, prob = prob)
  #print(table(y))
  data <- cbind(X, y = y)
  
  train_data <- data
  if(sum(train_data$y==0)==sum(train_data$y==1)){
    train_data_smote<-train_data
    train_data_smote1<-train_data
    train_data_kde <-train_data
    train_data_kde1 <-train_data
  }else{
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    n1<-sum(train_data$y==1)
    k1<-floor(n1^(4/(p+4)))
    
    m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
    train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
  }
  k_range <- floor(sqrt(NROW(train_data$y)))
  k_grid<-seq(5, 50, by = 1) 
  p_hat= mean(train_data$y==1)
  
  #Weighted model------------------------------------------------------------------
  weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )
  
  #Weighted CV--------
  train <- train_data[,-ncol(train_data)]
  trainlabels <- train_data$y
  results <- cross_validate_bknn(train=train, trainlabels=trainlabels, k_values =k_grid, m_folds=5)    
  results$best_k
  weighted_knn_cv <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=results$best_k, p_hat = p_hat )
  
  #logistic
  weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
  test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
  weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  #Smote-------------------------------
  #train_data_smote<- data_frame(train_data_smote)
  knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
  knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
  
  #logistic-----------------
  logistic_smote_fit <- glm(y ~ ., data = train_data_smote, family = "binomial")
  test_predictions <- predict(logistic_smote_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  logistic_smote_fit1 <- glm(y ~ ., data = train_data_smote1, family = "binomial")
  test_predictions <- predict(logistic_smote_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes1 <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_smote1 <- factor(predicted_classes1, levels = c(0, 1))  # Convert to factor
  
  #####
  k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
  
  train_data1<- train_data
  train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
  
  ##Smote applied at each fold a custom function for caret package when k=5 nearest neigbours
  smote_test <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = 5)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train <- train(y ~ ., 
                           data = train_data1, 
                           method = "knn",  # k-Nearest Neighbors classifier
                           trControl =  trainControl(method = "cv", 
                                                     number = 5,
                                                     classProbs = TRUE, 
                                                     summaryFunction = twoClassSummary 
                                                     ,sampling = smote_test),
                           metric = "ROC",  # Use ROC as the evaluation metric
                           tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors
  
  ##
  knn_smote_cv<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=knn_smote_train$bestTun$k)
  
  ##Smote applied at each fold a custom function for caret package when k=k1=n1^4/d+4 nearest neighbors
  smote_test_k1 <- list(
    name = "SMOTE with more neighbors!",
    func = function(x, y) {
      library(SMOTEWB)
      
      dat <- if (is.data.frame(x)) x else as.data.frame(x)
      dat$y <- y  # Ensure y is included in the data frame
      
      smote_result <- SMOTEWB::SMOTE(x = dat[,-ncol(dat)], 
                                     y = dat$y, 
                                     k = k1)
      
      # Ensure x is a data frame and y is a factor
      smote_data <- as.data.frame(smote_result$x)
      smote_data$y <- as.factor(smote_result$y)
      
      list(x = smote_result$x_new, y = as.factor(smote_result$y_new))
    },
    first = TRUE
  )
  
  ##
  knn_smote_train1 <- train(y ~ ., 
                            data = train_data1, 
                            method = "knn",  # k-Nearest Neighbors classifier
                            trControl =  trainControl(method = "cv", 
                                                      number = 5,
                                                      classProbs = TRUE, 
                                                      summaryFunction = twoClassSummary 
                                                      ,sampling = smote_test_k1),
                            metric = "ROC",  # Use ROC as the evaluation metric
                            tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  
  #
  knn_smote_cv1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=knn_smote_train1$bestTun$k)
  
  
  ##Kde-------------------------
  knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
  
  #logistic
  logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
  test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  
  
  ##Kde-------------------------
  knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
  
  #logistic
  logistic_kde_fit1 <- glm(y ~ ., data = train_data_kde1, family = "binomial")
  test_predictions <- predict(logistic_kde_fit1, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
  predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
  logistic_kde1 <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor
  ##
  ##Kde sampling applied at each fold a custom function for caret package with scott rule
  
  kde_test_scott <- list(name = "KDE with fold sampling!",
                         func = function (x,y) {
                           #library(DMwR)
                           dat <- if (is.data.frame(x)) x else as.data.frame(x)
                           dat <- balanced_data_kde(x=x,y=y, rule = "scott")
                           list(x = dat[,-ncol(dat)],
                                y = dat$y)
                         },
                         first = TRUE)
  
  
  
  knn_kde_train <- train(y ~ ., 
                         data = train_data1, 
                         method = "knn",  # k-Nearest Neighbors classifier
                         trControl =  trainControl(method = "cv", 
                                                   number = 5,
                                                   classProbs = TRUE, 
                                                   summaryFunction = twoClassSummary 
                                                   ,sampling = kde_test_scott),
                         metric = "ROC",  # Use ROC as the evaluation metric
                         tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  
  knn_kde_cv<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=knn_kde_train$bestTun$k)
  
  
  ##Kde sampling applied at each fold a custom function for caret package with smaller bandwidth e.g (scott_rule)/10
  
  kde_test_scott_else <- list(name = "KDE with fold sampling!",
                              func = function (x,y) {
                                #library(DMwR)
                                dat <- if (is.data.frame(x)) x else as.data.frame(x)
                                dat <- balanced_data_kde(x=x,y=y, rule = "scott_else")
                                list(x = dat[,-ncol(dat)],
                                     y = dat$y)
                              },
                              first = TRUE)
  
  
  
  knn_kde_train1 <- train(y ~ ., 
                          data = train_data1, 
                          method = "knn",  # k-Nearest Neighbors classifier
                          trControl =  trainControl(method = "cv", 
                                                    number = 5,
                                                    classProbs = TRUE, 
                                                    summaryFunction = twoClassSummary 
                                                    ,sampling = kde_test_scott_else),
                          metric = "ROC",  # Use ROC as the evaluation metric
                          tuneGrid=k_values)  # Tune for optimal 'k' (number of neighbors)
  
  
  knn_kde_cv1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=knn_kde_train1$bestTun$k)
  
  
  print(c( knn_smote_train$bestTun$k, knn_smote_train1$bestTun$k, knn_kde_train$bestTun$k,knn_kde_train1$bestTun$k))
  
  
  #balanced knn with AM risk------------------------------------------------------------------------------
  conf_weigted_knn_original<-confusionMatrix(weighted_knn_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_weigted_knn_cv<-confusionMatrix(weighted_knn_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv<-confusionMatrix(knn_smote_cv , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_smote_cv1<-confusionMatrix(knn_smote_cv1 , as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
  
  #Logistic with AM risk------------------------------------------------------------------------------
  conf_weigted_logistic_original<-confusionMatrix(weighted_logistic_original , as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote<-confusionMatrix(logistic_smote, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_smote1<-confusionMatrix(logistic_smote1, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv<-confusionMatrix(logistic_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
  #conf_logistic_smote_cv1<-confusionMatrix(logistic_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
  conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")
  
  #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
  
  
  #AM risk knn
  AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
  AM_knn_cv<-1-conf_weigted_knn_cv$byClass[11]
  AM_knn_smote<-1-conf_knn_smote$byClass[11]
  AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
  AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
  AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
  AM_knn_kde<-1-conf_knn_kde$byClass[11]
  AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
  AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
  AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
  
  #
  AM_knn<- c(AM_knn_original,AM_knn_cv, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
  names(AM_knn)<-  c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")
  #print(AM_knn)
  
  # Save risk in the appropriate matrix
  
  
  
  #AM risk logistic
  AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
  AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
  AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
  AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
  AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
  #
  AM_logistic<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
  names(AM_logistic)<-     c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
  # Save risk in the appropriate matrix
  
  # Save risk in the appropriate matrix
  AM_risk_knn$alpha_06[r, ] <- AM_knn
  AM_risk_logistic$alpha_06[r, ] <- AM_logistic
  
  
}
close(pb)



# Mean AM risk knn------------------------------------
AM_risk_knn_alpha_01<- round(colMeans(na.omit(AM_risk_knn$alpha_01)), 4)
AM_risk_knn_alpha_02<- round(colMeans(na.omit(AM_risk_knn$alpha_02)), 4)
AM_risk_knn_alpha_03<- round(colMeans(na.omit(AM_risk_knn$alpha_03)), 4)
AM_risk_knn_alpha_04<- round(colMeans(na.omit(AM_risk_knn$alpha_04)), 4)
AM_risk_knn_alpha_05<- round(colMeans(na.omit(AM_risk_knn$alpha_05)), 4)
AM_risk_knn_alpha_06<- round(colMeans(na.omit(AM_risk_knn$alpha_06)), 4)




# Mean AM risk logistic------------------------------------
AM_risk_logistic_alpha_01<- round(colMeans(na.omit(AM_risk_logistic$alpha_01)), 4)
AM_risk_logistic_alpha_02<- round(colMeans(na.omit(AM_risk_logistic$alpha_02)), 4)
AM_risk_logistic_alpha_03<- round(colMeans(na.omit(AM_risk_logistic$alpha_03)), 4)
AM_risk_logistic_alpha_04<- round(colMeans(na.omit(AM_risk_logistic$alpha_04)), 4)
AM_risk_logistic_alpha_05<- round(colMeans(na.omit(AM_risk_logistic$alpha_05)), 4)
AM_risk_logistic_alpha_06<- round(colMeans(na.omit(AM_risk_logistic$alpha_06)), 4)



#ggsave("ECSD-exp-gpd-kde-nn.pdf", plot = p, width = 5, height = 5, dpi = 300)##AM_risk_standard_knn_with_balanced_test------------------------------

library(ggplot2)

# Define the methods and labels
methods <- c("-1","-1.5","-2", "-2.5", "-3", "-3.5")
labels <- c("BBC","BBC-CV", "SMOTE(S)","SMOTE(L)", "SMOTE(S)_CV","SMOTE(L)_CV","KDE(L)", "KDE(L)-CV", "KDE(S)", "KDE(S)-CV")

##AM_risk_standard_knn_with_imbalanced_test----------

dfs <- lapply(1:10, function(i) {
  data.frame(
    d = rep(labels[i], 6),
    method = methods,
    x = c(
      AM_risk_knn_alpha_01[i],
      AM_risk_knn_alpha_02[i],
      AM_risk_knn_alpha_03[i],
      AM_risk_knn_alpha_04[i],
      AM_risk_knn_alpha_05[i],
      AM_risk_knn_alpha_06[i]
    )
  )
})


# Combine all data frames
AM_risk_KNN <- do.call(rbind, dfs)

# Convert "method" to ordered factor with custom levels
AM_risk_KNN$method <- factor(
  AM_risk_KNN$method,
  levels = methods,
  ordered = TRUE
)

# Reorder the levels of "d" variable
AM_risk_KNN$d <- factor(
  AM_risk_KNN$d,
  levels = labels
)
##


AM_risk_KNN$t <-  "Example 1"
##

manual_shapes <- c(
  "BBC" = 4,        # cross
  "BBC-CV" = 16,    # dot
  "SMOTE(S)" = 16,      # cross
  "SMOTE(L)" = 4,     # dot
  "SMOTE(S)-CV" = 16,
  "SMOTE(L)-CV" = 4,
  "KDE(L)" = 4,
  "KDE(S)" = 16,
  "KDE(L)-CV" = 4,
  "KDE(S)-CV" = 16
)

# Define custom colors (same for related methods)
manual_colors <- c(
  "BBC" = "#1b9e77",       # greenish
  "BBC-CV" = "#1b9e77",    
  "SMOTE(S)" = "#d95f02",     # orange
  "SMOTE(L)" = "#d95f02",     
  "SMOTE(S)-CV" = "#7570b3",  # purple
  "SMOTE(L)-CV" = "#7570b3",  
  "KDE(L)" = "#e7298a",            # pink
  "KDE(S)" = "#e7298a",
  "KDE(L)-CV" = "#66a61e",         # green
  "KDE(S)-CV" = "#66a61e"
)



p <- ggplot(AM_risk_KNN, aes(x = method, y = x, group = d, color = d, shape = d)) +
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

ggsave("AM-Exm1-KNN.pdf", plot = p, width = 6, height = 4, dpi = 300)##AM_risk_standard_knn_with_balanced_test------------------------------




##Logistic case


labels <-  c("BBC","SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")

## AM_risk_balanced_knn_with_imbalanced_test----------------
dfs <- lapply(1:5, function(i) {
  data.frame(
    d = rep(labels[i], 6),
    method = methods,
    x = c(
      AM_risk_logistic_alpha_01[i],
      AM_risk_logistic_alpha_02[i],
      AM_risk_logistic_alpha_03[i],
      AM_risk_logistic_alpha_04[i],
      AM_risk_logistic_alpha_05[i],
      AM_risk_logistic_alpha_06[i]
    )
  )
})

# Combine all data frames
AM_risk_LOGISTIC<- do.call(rbind, dfs)

# Convert "method" to ordered factor with custom levels
AM_risk_LOGISTIC$method <- factor(
  AM_risk_LOGISTIC$method,
  levels = methods,
  ordered = TRUE
)

# Reorder the levels of "d" variable
AM_risk_LOGISTIC$d <- factor(
  AM_risk_LOGISTIC$d,
  levels = labels
)
AM_risk_LOGISTIC$t <-  "Example 1"
##

manual_shapes <- c(
  "BBC" = 4,        # cross
 # "BBC-CV" = 16,    # dot
  "SMOTE(S)" = 16,      # cross
  "SMOTE(L)" = 4,     # dot
  #"SMOTE(S)-CV" = 16,
 # "SMOTE(L)-CV" = 4,
  "KDE(L)" = 4,
  "KDE(S)" = 16
  #"KDE(L)-CV" = 4,
  #"KDE(S)-CV" = 16
)

# Define custom colors (same for related methods)
manual_colors <- c(
  "BBC" = "#1b9e77",       # greenish
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



p <- ggplot(AM_risk_LOGISTIC, aes(x = method, y = x, group = d, color = d, shape = d)) +
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

ggsave("AM-Exm1-logis.pdf", plot = p, width = 6, height = 4, dpi = 300)##AM_risk_standard_knn_with_balanced_test------------------------------



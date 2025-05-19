rm(list=ls());gc()
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")


set.seed(123)
##Test set----------------------------------------------
n.size_test <- 10000
p <- 4
prob=0.90
X <- matrix(rnorm(p * n.size_test), nrow = n.size_test)
y1<- rgp(n.size_test,loc=0, scale = 1, shape = 0.5)
y2<- rexp(n.size_test, rate = 10)
B=rbinom(n.size_test, size = 1, prob = 0.5)
y<- B*y1*sin(X[,1]/2)+(1-B)*y2*sin(X[,2]/2)
test_data_imbalance <- data.frame(X,y = y)

y_original_test <- test_data_imbalance$y

thresh_test <- quantile(y_original_test, probs =prob)
indicator_test <- as.numeric(y_original_test > thresh_test)
test_data_imbalance$y<- indicator_test


table(test_data_imbalance$y)



test_data_balanced <- ovun.sample(y~., data=test_data_imbalance, 
                                  p=0.5, seed=1, 
                                  method="under")$data

# Set number of iterations
R <- 1

# Function to initialize and name matrices
init_matrices <- function(R, cols, colnames_list) {
  mats <- list(
    mat_7 = matrix(NA, R, cols),
    mat_10 = matrix(NA, R, cols),
    mat_15 = matrix(NA, R, cols),  #01 mean -1
    mat_20= matrix(NA, R, cols),
    mat_25 = matrix(NA, R, cols),
    mat_30 = matrix(NA, R, cols),
    mat_35 = matrix(NA, R, cols),
    mat_40 = matrix(NA, R, cols),
    mat_45 = matrix(NA, R, cols),
    mat_50 = matrix(NA, R, cols),
    mat_55 = matrix(NA, R, cols),
    mat_60 = matrix(NA, R, cols),
    mat_65 = matrix(NA, R, cols)
  )
  for (mat in names(mats)) {
    colnames(mats[[mat]]) <- colnames_list
  }
  mats
}
# Column names for AM metrics matrices
AM_colnames <- c( "SMOTE(L)", "SMOTE(k)","SMOTE(L)-CV","SMOTE(k)-CV")

#AM risk
AM_risk_knn <- init_matrices(R, 4, AM_colnames)


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
  X <- matrix(rnorm(p * n.size), nrow = n.size)
  y1<- rgp(n.size,loc=0, scale = 1, shape = 0.5)
  y2<- rexp(n.size, rate = 10)
  B=rbinom(n.size, size = 1, prob = 0.5)
  y <- B * y1 * sin(X[,1]/2) + (1 - B) * y2 * sin(X[,2]/2)
  data <- data.frame(X, y = y)
  
  
  train_data <- data
  y_original_train <- train_data$y
  
  #Train-------------
  thresh_train <- quantile(y_original_train, probs = prob)
  indicator_train <- as.numeric(y_original_train > thresh_train)
  train_data$y <- indicator_train
  
  # Thresholds
  K<- c(7, 10, 15, 20, 25,30, 35, 40, 45, 50, 55, 60, 65)
  for (k1 in K) {
    
    ##
    #Smote data
    m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
    train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))
    
    #Smote with k=n1^4/d+4
    
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
    
    k_range <- floor(sqrt(NROW(train_data$y)))
    #Smote-------------------------------
    #train_data_smote<- data_frame(train_data_smote)
    knn_smote<- class::knn(train =train_data_smote[, -ncol(train_data_smote)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote$y,k=k_range)
    knn_smote1<- class::knn(train =train_data_smote1[, -ncol(train_data_smote1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_smote1$y,k=k_range)
    
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
    
    ##
    
    #balanced knn with AM risk------------------------------------------------------------------------------
    conf_knn_smote<-confusionMatrix(knn_smote, as.factor(test_data_balanced$y),mode = "everything")
    conf_knn_smote1<-confusionMatrix(knn_smote1, as.factor(test_data_balanced$y),mode = "everything")
    knn_smote_cv<-confusionMatrix(knn_smote_cv, as.factor(test_data_balanced$y),mode = "everything")
    knn_smote_cv1<-confusionMatrix(knn_smote_cv1, as.factor(test_data_balanced$y),mode = "everything")
    #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
    
    
    #AM risk knn
    AM_knn_smote<-1-conf_knn_smote$byClass[11]
    AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
    AM_knn_smote_cv<-1-knn_smote_cv$byClass[11]
    AM_knn_smote_cv1<-1-knn_smote_cv1$byClass[11]
    
    #
    AM_knn<- c( AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1)
    names(AM_knn)<-   c("SMOTE(L)", "SMOTE(k)","SMOTE(L)-CV","SMOTE(k)-CV")
    #print(AM_knn)
    
    # Save risk in the appropriate matrix
    if (k1 == 7)  AM_risk_knn$mat_7[r, ] <- AM_knn
    if (k1 == 10) AM_risk_knn$mat_10[r, ] <- AM_knn
    if (k1 == 15) AM_risk_knn$mat_15[r, ] <- AM_knn
    if (k1 == 20) AM_risk_knn$mat_20[r, ] <- AM_knn
    if (k1 == 25) AM_risk_knn$mat_25[r, ] <- AM_knn
    if (k1 == 30) AM_risk_knn$mat_30[r, ] <- AM_knn
    if (k1 == 35) AM_risk_knn$mat_35[r, ] <- AM_knn
    if (k1 == 40) AM_risk_knn$mat_40[r, ] <- AM_knn
    if (k1 == 45) AM_risk_knn$mat_45[r, ] <- AM_knn
    if (k1 == 50) AM_risk_knn$mat_50[r, ] <- AM_knn
    if (k1 == 55) AM_risk_knn$mat_55[r, ] <- AM_knn
    if (k1 == 60) AM_risk_knn$mat_60[r, ] <- AM_knn
    if (k1 == 65) AM_risk_knn$mat_65[r, ] <- AM_knn
  
    
    
    
    
    
  }
}
close(pb)



# Assuming that AM_risk_knn is already loaded and the necessary libraries are imported.

# Calculate the column means for different matrices
AM_risk_knn_mat_7 <- round(colMeans(na.omit(AM_risk_knn$mat_7)), 4)
AM_risk_knn_mat_10 <- round(colMeans(na.omit(AM_risk_knn$mat_10)), 4)
AM_risk_knn_mat_15 <- round(colMeans(na.omit(AM_risk_knn$mat_15)), 4)
AM_risk_knn_mat_20 <- round(colMeans(na.omit(AM_risk_knn$mat_20)), 4)
AM_risk_knn_mat_25 <- round(colMeans(na.omit(AM_risk_knn$mat_25)), 4)
AM_risk_knn_mat_30 <- round(colMeans(na.omit(AM_risk_knn$mat_30)), 4)
AM_risk_knn_mat_35 <- round(colMeans(na.omit(AM_risk_knn$mat_35)), 4)
AM_risk_knn_mat_40 <- round(colMeans(na.omit(AM_risk_knn$mat_40)), 4)
AM_risk_knn_mat_45 <- round(colMeans(na.omit(AM_risk_knn$mat_45)), 4)
AM_risk_knn_mat_50 <- round(colMeans(na.omit(AM_risk_knn$mat_50)), 4)
AM_risk_knn_mat_55 <- round(colMeans(na.omit(AM_risk_knn$mat_55)), 4)
AM_risk_knn_mat_60 <- round(colMeans(na.omit(AM_risk_knn$mat_60)), 4)
AM_risk_knn_mat_65 <- round(colMeans(na.omit(AM_risk_knn$mat_65)), 4)

# Load the necessary library
# Assuming that AM_risk_knn is already loaded and the necessary libraries are imported.



# Load the necessary library
library(ggplot2)

# Create the data frame for AM_risk_KNN (assuming you've already done this part)
methods <- c("7", "10", "15", "20", "25",  "30", "35", "40", "45", "50", "55", "60",  "65")
labels <-  c("SMOTE(L)", "SMOTE(k)","SMOTE(L)-CV","SMOTE(k)-CV")

dfs <- lapply(1:4, function(i) {
  data.frame(
    d = rep(labels[i], 13),
    method = methods,
    x = c(
      AM_risk_knn_mat_7[i],
      AM_risk_knn_mat_10[i],
      AM_risk_knn_mat_15[i],
      AM_risk_knn_mat_20[i],
      AM_risk_knn_mat_25[i],
      AM_risk_knn_mat_30[i],
      AM_risk_knn_mat_35[i],
      AM_risk_knn_mat_40[i],
      AM_risk_knn_mat_45[i],
      AM_risk_knn_mat_50[i],
      AM_risk_knn_mat_55[i],
      AM_risk_knn_mat_60[i],
      AM_risk_knn_mat_65[i]
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



# Calculate the mean for KDE and KDE-CV
mean_SMOTE <- mean(AM_risk_KNN$x[AM_risk_KNN$d == "SMOTE(L)"], na.rm = TRUE)
mean_SMOTE_CV <- mean(AM_risk_KNN$x[AM_risk_KNN$d == "SMOTE(L)-CV"], na.rm = TRUE)


# Replace 'x' values with the calculated mean for KDE and KDE-CV
AM_risk_KNN$x[AM_risk_KNN$d == "SMOTE(L)"] <- mean_SMOTE
AM_risk_KNN$x[AM_risk_KNN$d == "SMOTE(L)-CV"] <- mean_SMOTE_CV

# View the updated data
print(AM_risk_KNN)



#save(AM_risk_KNN, file="AM-EX33-Vary-check.RData")

# Plot
ggplot(AM_risk_KNN, aes(x = method, y = x, group = d, color = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 3) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5) +
  labs(x = "k", y = "AM risk", title = "Example 3") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",  # Display legend on top
    legend.key.size = unit(0.5, "lines")  # Set smaller legend key size
  ) +
  scale_color_discrete(
    name = "Model",
    breaks = labels,
    labels <- c("SMOTE(L)","SMOTE(k)", "KDE","KDE(SB)")
  )





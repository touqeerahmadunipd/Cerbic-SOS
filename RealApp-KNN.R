rm(list=ls());gc()
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")
###################################
##  phoneme data   ###############
#################################
set.seed(123)
data<- read.csv("phoneme.csv")
head(data)
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
##Balane the test set
test_data_balanced <- ovun.sample(y~., data=test_data, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)

#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))

#print(table(train_data_kde$y))
###KNN--------------------------------
train_data$y <- as.numeric(as.character(train_data$y))

#fixed k choice
k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )

#logistic
weighted_logistic_original_fit <- glm(y ~ ., data = train_data, family = "binomial")
test_predictions <- predict(weighted_logistic_original_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
predicted_classes <- ifelse(test_predictions > p_hat, 1, 0)  # Convert probabilities to class labels (0 or 1)
weighted_logistic_original <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor

#Smote-------------------------------

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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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

##Kde with H-------------------------
knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)

#logistic
logistic_kde_fit <- glm(y ~ ., data = train_data_kde, family = "binomial")
test_predictions <- predict(logistic_kde_fit, newdata = test_data_balanced[,- ncol(test_data_balanced)], type = "response")
predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
logistic_kde <- factor(predicted_classes, levels = c(0, 1))  # Convert to factor


##Kde with smaller H-------------------------
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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_phoneme<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_phoneme)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_phoneme<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_phoneme)<-     c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix



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

# Check dimensions
dim(train_data)
dim(test_data)
table(train_data$y)
##Balane the test set
test_data_balanced <- ovun.sample(y~., data=test_data, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)






#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))

#print(table(train_data_kde$y))
###KNN--------------------------------
train_data$y <- as.numeric(as.character(train_data$y))

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_house_16H<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_house_16H)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_house_16H<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_house_16H)<-    c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")











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

set.seed(123)



# Split the data (e.g., 80% training, 20% testing)
split <- sample.split(data$y, SplitRatio = 0.7)  # Assuming "Class" is the target variable

# Create training and testing sets
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)





##Imblancedness
# Set seed for reproducibility


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

# Check class distributions
table(train_data_20$y)
table(train_data_10$y)
table(train_data_05$y)


train_data<-train_data_20


# Check dimensions
dim(train_data)
dim(test_data)
table(train_data$y)
table(test_data$y)
#Test set is already balanced
test_data_balanced<- test_data


#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))


train_data$y <- as.numeric(as.character(train_data$y))
#print(table(train_data_kde$y))
###KNN--------------------------------

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
#knn_Weighted_standard_with_imbalanced_test<- class::knn(train = train_data[, -ncol(train_data)], test = test_data_balanced[,- ncol(test_data_balanced)] ,cl =train_data$y,k=k_range)
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )

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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_california_0.20<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_california_0.20)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_california_0.20<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_california_0.20)<-    c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix





### At 10% imbalanced ration


train_data<-train_data_10
test_data_balanced<- test_data





#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))



train_data$y <- as.numeric(as.character(train_data$y))
#print(table(train_data_kde$y))
###KNN--------------------------------

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")

#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_california_0.10<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_california_0.10)<-    c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_california_0.10<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_california_0.10)<-    c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")







##At 5% imbalanced ratio--------------------------------------------------------

train_data<-train_data_05
test_data_balanced<- test_data


#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))

#print(table(train_data_kde$y))
###KNN--------------------------------
train_data$y <- as.numeric(as.character(train_data$y)) 

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_california_0.05<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_california_0.05)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_california_0.05<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_california_0.05)<-    c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")







#########################################
###  MagicTEl  from OpenML  ###############---------------------------------------
#########################################

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
# Create datasets with different minority class ratios
train_data_20 <- create_imbalance(train_data, 0.20)
train_data_10 <- create_imbalance(train_data, 0.10)
train_data_05  <- create_imbalance(train_data, 0.05)

# Check class distributions
table(train_data_20$y)
table(train_data_10$y)
table(train_data_05$y)

# 20% imbalanced ratio-----------------------------------------------------
train_data<-train_data_20
test_data_balanced<- test_data



#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))


train_data$y <- as.numeric(as.character(train_data$y)) 
#print(table(train_data_kde$y))
###KNN--------------------------------

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
#knn_Weighted_standard_with_imbalanced_test<- class::knn(train = train_data[, -ncol(train_data)], test = test_data_balanced[,- ncol(test_data_balanced)] ,cl =train_data$y,k=k_range)
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_MagicTel_0.20<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_MagicTel_0.20)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_MagicTel_0.20<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_MagicTel_0.20)<-     c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix









# 10% imbalanced ratio-----------------------------------------------------
train_data<-train_data_10
test_data_balanced<- test_data




#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))

#print(table(train_data_kde$y))
###KNN--------------------------------

train_data$y <- as.numeric(as.character(train_data$y)) 
k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )

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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_MagicTel_0.10<- c(AM_knn_original,AM_knn_smote, AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_MagicTel_0.10)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_MagicTel_0.10<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_MagicTel_0.10)<-     c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix















# 5% imbalanced ratio-----------------------------------------------------
train_data<-train_data_05
test_data_balanced<- test_data




#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))


train_data$y <- as.numeric(as.character(train_data$y)) 
#print(table(train_data_kde$y))
###KNN--------------------------------

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_MagicTel_0.05<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_MagicTel_0.05)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]
#
AM_logistic_MagicTel_0.05<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_MagicTel_0.05)<-     c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix



###############################
##    abalone data  ##########
##############################
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
##Balanced the test set
test_data_balanced <- ovun.sample(y~., data=test_data, 
                                  p=0.5, seed=1, 
                                  method="under")$data

head(test_data_balanced)
table(test_data_balanced$y)



#Smote data
m_train <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = 5)
train_data_smote <- data.frame(m_train$x_new, y=as.numeric(as.character(m_train$y_new)))

#Smote with k=n1^4/d+4
n1<-sum(train_data$y==1)
d=ncol(train_data)-1
k1<-floor(n1^(4/(d+4)))

m_train1 <- SMOTEWB::SMOTE(x = train_data[,-ncol(train_data)], y = as.factor(train_data$y), k = k1)
train_data_smote1 <- data.frame(m_train1$x_new, y=as.numeric(as.character(m_train1$y_new)))


#kde train set-------------------------
train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
train_data_kde$y<- as.numeric(as.character(train_data_kde$y))

#kde train set with smaller bandwidth-------------------------
train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else")
train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))

#print(table(train_data_kde$y))
###KNN--------------------------------
train_data$y <- as.numeric(as.character(train_data$y))

k_range <- floor(sqrt(NROW(train_data$y)))
p_hat= mean(train_data$y==1)

#Weighted model------------------------------------------------------------------
weighted_knn_original <- balanced_knn(train=train_data[, -ncol(train_data)], trainlabels =  train_data$y, test = test_data_balanced[,- ncol(test_data_balanced)], k=k_range, p_hat = p_hat )


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
k_values <- expand.grid(k = seq(3, 200, by=2))  # Ensure the column name is 'k'

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
                         dat <- balanced_data_kde1(x=x,y=y, rule = "scott")
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
                              dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else")
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
conf_logistic_kde<-confusionMatrix(logistic_kde, as.factor(test_data_balanced$y),mode = "everything")
conf_logistic_kde1<-confusionMatrix(logistic_kde1, as.factor(test_data_balanced$y),mode = "everything")



#AM risk knn
AM_knn_original<-1-conf_weigted_knn_original$byClass[11]
AM_knn_smote<-1-conf_knn_smote$byClass[11]
AM_knn_smote1<-1-conf_knn_smote1$byClass[11]
AM_knn_smote_cv<-1-conf_knn_smote_cv$byClass[11]
AM_knn_smote_cv1<-1-conf_knn_smote_cv1$byClass[11]
AM_knn_kde<-1-conf_knn_kde$byClass[11]
AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]

#
AM_knn_abalone<- c(AM_knn_original, AM_knn_smote,AM_knn_smote1, AM_knn_smote_cv,AM_knn_smote_cv1, AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv, AM_knn_kde_cv1)
names(AM_knn_abalone)<-   c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV")
#print(AM_knn)


#AM risk logistic
AM_logistic_original<-1-conf_weigted_logistic_original$byClass[11]
AM_logistic_smote<-1-conf_logistic_smote$byClass[11]
AM_logistic_smote1<-1-conf_logistic_smote1$byClass[11]
AM_logistic_kde<-1-conf_logistic_kde$byClass[11]
AM_logistic_kde1<-1-conf_logistic_kde1$byClass[11]v 
#
AM_logistic_abalone<- c(AM_logistic_original,AM_logistic_smote,AM_logistic_smote1,  AM_logistic_kde,  AM_logistic_kde1)
names(AM_logistic_abalone)<-     c("BBC", "SMOTE(S)","SMOTE(L)","KDE(L)","KDE(S)")
# Save risk in the appropriate matrix










# Load required library
library(ggplot2)

# Define methods (each dataset has these 5 methods)
methods <- rep(c("BBC", "SMOTE(S)","SMOTE(L)", "SMOTE(S)-CV","SMOTE(L)-CV","KDE(L)", "KDE(S)", "KDE(L)-CV",  "KDE(S)-CV"), 9)

# Define dataset names (each appears 5 times)
datasets <- rep(c("Phoneme","Abalone", "House_16H", "California(5%)", "California(10%)", 
                  "California(20%)", "MagicTel(5%)", "MagicTel(10%)", "MagicTel(20%)"), each = 9)


AM_values <- c(
  AM_knn_phoneme[1:9],
  AM_knn_abalone[1:9],
  AM_knn_house_16H[1:9],
  AM_knn_california_0.05[1:9],
  AM_knn_california_0.10[1:9],
  AM_knn_california_0.20[1:9],
  AM_knn_MagicTel_0.05[1:9],
  AM_knn_MagicTel_0.10[1:9],
  AM_knn_MagicTel_0.20[1:9]
)
# Create a data frame
data <- data.frame(Methods = methods, Dataset = datasets, AM_Risk = AM_values)


library(dplyr)

# Ensure Dataset and Methods are factors
data$Dataset <- factor(data$Dataset, 
                       levels = c("Abalone","California(20%)", "California(10%)", "California(5%)", 
                                  "MagicTel(20%)", "MagicTel(10%)", "MagicTel(5%)", 
                                  "Phoneme", "House_16H"))

desired_order <- c("BBC", "KDE(S)", "SMOTE(S)", "KDE(L)", "SMOTE(L)",
                   "KDE-CV(S)", "SMOTE-CV(S)", "KDE(L)-CV", "SMOTE(L)-CV")

# Reorder methods by desired order within each dataset
data_ordered <- data %>%
  mutate(Methods = factor(Methods, levels = desired_order)) %>%
  arrange(Dataset, Methods)

# View result
print(data_ordered)

data=data_ordered

library(ggplot2)
library(dplyr)
save(data, file="AM-risk-realapp_knn.RData")
load("AM-risk-realapp_knn.RData")
# Flag minimum values per dataset
data_annotated <- data %>%
  group_by(Dataset) %>%
  mutate(is_min = AM_Risk == min(AM_Risk))

# Define manual colors (adjust to your preferred palette)


manual_colors <- c(
  "BBC"        = "#E63946",  # strong red
  "KDE(S)"         = "#457B9D",  # cool blue
  "SMOTE(S)"      = "#2A9D8F",  # teal green
  "KDE(L)"             = "#F4A261",  # warm orange
  "SMOTE(L)"      = "#A8DADC",  # pale cyan
  "KDE(S)-CV"      = "blue",  # soft gray-blue
  "SMOTE(S)-CV"   = "#F72585",  # hot pink
  "KDE(L)-CV"          = "#7209B7",  # deep violet
  "SMOTE(L)-CV"   = "#06D6A0"   # mint green
)


data_annotated <- data_annotated[data_annotated$Methods != "BBC", ]

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
  labs(title = "AM Risk by Different Methods for Different Datasets Using kNN",
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


p <- ggplot(data_annotated, aes(x = Dataset, y = AM_Risk, fill = Methods)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  labs(title = "AM Risk when using KNN",
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




ggsave("Real-data-plot_knn.pdf", plot = p, width = 6, height = 6, dpi = 300)##AM_risk_standard_knn_with_balanced_test------------------------------

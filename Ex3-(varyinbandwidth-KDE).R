rm(list=ls());gc()
setwd("D:/PostDoc work/code/Main code/LLOX/After-Pak-visit/Cleaned Code")
source("Functions.R")







balanced_data_kde1 <- function(x, y, rule=c("scott","scott_else"), k=NA) {
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
    synthetic_minority_samples <- kernelboot::rmvg(num_synthetic_samples, minority_data, bw =(k)*bw.scott(minority_data), weights = NULL, adjust = 1)
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
R <- 50

# Function to initialize and name matrices
init_matrices <- function(R, cols, colnames_list) {
  mats <- list(
    mat_20 = matrix(NA, R, cols),
    mat_10 = matrix(NA, R, cols),
    mat_5 = matrix(NA, R, cols),  #01 mean -1
    mat_3= matrix(NA, R, cols),
    mat_2 = matrix(NA, R, cols),
    mat_1.5 = matrix(NA, R, cols),
    mat_0.6 = matrix(NA, R, cols),
    mat_0.5 = matrix(NA, R, cols),
    mat_0.3 = matrix(NA, R, cols),  #01 mean -1
    mat_0.2= matrix(NA, R, cols),
    mat_0.1 = matrix(NA, R, cols),
    mat_0.05 = matrix(NA, R, cols)
  )
  for (mat in names(mats)) {
    colnames(mats[[mat]]) <- colnames_list
  }
  mats
}
# Column names for AM metrics matrices
AM_colnames <- c( "KDE", "KDE(SB)","KDE-CV","SMOTE(SB)-CV")
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
  
  # Bandwidth
  K<- c( 20, 10, 5, 3, 2, 1.5, 0.6666667, 0.5, 0.3333333, 0.2, 0.1, 0.05)
  for (k1 in K) {
    
    
    #kde train set-------------------------
    train_data_kde <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott")
    train_data_kde$y<- as.numeric(as.character(train_data_kde$y))
    
    #kde train set with smaller bandwidth-------------------------
    train_data_kde1 <- balanced_data_kde1(train_data[, -ncol(train_data)], y=train_data$y, rule = "scott_else", k=k1)
    train_data_kde1$y<- as.numeric(as.character(train_data_kde1$y))
    #print(table(train_data_kde$y))
    ###KNN--------------------------------
    k_range <- floor(sqrt(NROW(train_data$y)))
   
    
    knn_kde<- class::knn(train =train_data_kde[, -ncol(train_data_kde)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde$y,k=k_range)
    
    knn_kde1<- class::knn(train =train_data_kde1[, -ncol(train_data_kde1)], test =test_data_balanced[,- ncol(test_data_balanced)],cl =train_data_kde1$y,k=k_range)
    
    
    #####
    k_values <- expand.grid(k = seq(2, 300, by=5))  # Ensure the column name is 'k'
    
    train_data1<- train_data
    train_data1$y <- as.factor(ifelse(train_data1$y == 0, "class0", "class1"))
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
                                  dat <- balanced_data_kde1(x=x,y=y, rule = "scott_else", k=k1)
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
    
    
    ##
    
    #balanced knn with AM risk------------------------------------------------------------------------------
    conf_knn_kde<-confusionMatrix(knn_kde, as.factor(test_data_balanced$y),mode = "everything")
    conf_knn_kde1<-confusionMatrix(knn_kde1, as.factor(test_data_balanced$y),mode = "everything")
    conf_knn_kde_cv<-confusionMatrix(knn_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
    conf_knn_kde_cv1<-confusionMatrix(knn_kde_cv1, as.factor(test_data_balanced$y),mode = "everything")
    #conf_logistic_kde_cv<-confusionMatrix(logistic_kde_cv, as.factor(test_data_balanced$y),mode = "everything")
    
    
    #AM risk knn
    AM_knn_kde<-1-conf_knn_kde$byClass[11]
    AM_knn_kde1<-1-conf_knn_kde1$byClass[11]
    AM_knn_kde_cv<-1-conf_knn_kde_cv$byClass[11]
    AM_knn_kde_cv1<-1-conf_knn_kde_cv1$byClass[11]
    
    #
    AM_knn<- c( AM_knn_kde,AM_knn_kde1, AM_knn_kde_cv,AM_knn_kde_cv1)
    names(AM_knn)<-   c("KDE", "KDE(SB)","KDE-CV","SMOTE(SB)-CV")
    print(AM_knn)
    
    # Save risk in the appropriate matrix
    if (k1 == 20)  AM_risk_knn$mat_20[r, ] <- AM_knn
    if (k1 == 10)  AM_risk_knn$mat_10[r, ] <- AM_knn
    if (k1 == 5) AM_risk_knn$mat_5[r, ] <- AM_knn
    if (k1 == 3) AM_risk_knn$mat_3[r, ] <- AM_knn
    if (k1 == 2) AM_risk_knn$mat_2[r, ] <- AM_knn
    if (k1 == 1.5) AM_risk_knn$mat_1.5[r, ] <- AM_knn
    if (k1 == 0.6666667)  AM_risk_knn$mat_0.6[r, ] <- AM_knn
    if (k1 == 0.5)  AM_risk_knn$mat_0.5[r, ] <- AM_knn
    if (k1 == 0.3333333) AM_risk_knn$mat_0.3[r, ] <- AM_knn
    if (k1 == 0.2) AM_risk_knn$mat_0.2[r, ] <- AM_knn
    if (k1 == 0.1) AM_risk_knn$mat_0.1[r, ] <- AM_knn
    if (k1 == 0.05) AM_risk_knn$mat_0.05[r, ] <- AM_knn
    
    
    
    
  }
}
close(pb)


# Assuming that AM_risk_knn is already loaded and the necessary libraries are imported.

# Calculate the column means for different matrices
AM_risk_knn_mat_20 <- round(colMeans(na.omit(AM_risk_knn$mat_20)), 4)
AM_risk_knn_mat_10 <- round(colMeans(na.omit(AM_risk_knn$mat_10)), 4)
AM_risk_knn_mat_5 <- round(colMeans(na.omit(AM_risk_knn$mat_5)), 4)
AM_risk_knn_mat_3 <- round(colMeans(na.omit(AM_risk_knn$mat_3)), 4)
AM_risk_knn_mat_2 <- round(colMeans(na.omit(AM_risk_knn$mat_2)), 4)
AM_risk_knn_mat_1.5 <- round(colMeans(na.omit(AM_risk_knn$mat_1.5)), 4)
AM_risk_knn_mat_0.05 <- round(colMeans(na.omit(AM_risk_knn$mat_0.05)), 4)
AM_risk_knn_mat_0.1 <- round(colMeans(na.omit(AM_risk_knn$mat_0.1)), 4)
AM_risk_knn_mat_0.2 <- round(colMeans(na.omit(AM_risk_knn$mat_0.2)), 4)
AM_risk_knn_mat_0.3 <- round(colMeans(na.omit(AM_risk_knn$mat_0.3)), 4)
AM_risk_knn_mat_0.5 <- round(colMeans(na.omit(AM_risk_knn$mat_0.5)), 4)
AM_risk_knn_mat_0.6 <- round(colMeans(na.omit(AM_risk_knn$mat_0.6)), 4)

# Load the necessary library
library(ggplot2)

# Create the data frame for AM_risk_KNN (assuming you've already done this part)
#methods <- c("20H", "10H", "5H", "3H", "2H", "1.5H", "H/20", "H/10", "H/5", "H/3", "H/2","H/1.5")
methods <- c( "H/20", "H/10", "H/5", "H/3", "H/2","H/1.5" , "1.5H", "2H", "3H")
labels <- c("KDE", "KDE(VB)", "KDE-CV", "KDE(VB)-CV")

dfs <- lapply(1:4, function(i) {
  data.frame(
    d = rep(labels[i], 9),
    method = methods,
    x = c(
      # AM_risk_knn_mat_20[i],
      # AM_risk_knn_mat_10[i],
      # AM_risk_knn_mat_5[i],
      
      AM_risk_knn_mat_0.05[i],
      AM_risk_knn_mat_0.1[i],
      AM_risk_knn_mat_0.2[i],
      AM_risk_knn_mat_0.3[i],
      AM_risk_knn_mat_0.5[i],
      AM_risk_knn_mat_0.6[i], 
      AM_risk_knn_mat_1.5[i],
      AM_risk_knn_mat_2[i],
      AM_risk_knn_mat_3[i]
      
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
mean_KDE <- mean(AM_risk_KNN$x[AM_risk_KNN$d == "KDE"], na.rm = TRUE)
mean_KDE_CV <- mean(AM_risk_KNN$x[AM_risk_KNN$d == "KDE-CV"], na.rm = TRUE)

# Print the means
print(paste("Mean for KDE: ", mean_KDE))
print(paste("Mean for KDE-CV: ", mean_KDE_CV))

# Replace 'x' values with the calculated mean for KDE and KDE-CV
AM_risk_KNN$x[AM_risk_KNN$d == "KDE"] <- mean_KDE
AM_risk_KNN$x[AM_risk_KNN$d == "KDE-CV"] <- mean_KDE_CV

# View the updated data
print(AM_risk_KNN)



save(AM_risk_KNN, file="AM-EX3-bandwidthvary-check.RData")
# Plot
ggplot(AM_risk_KNN, aes(x = method, y = x, group = d, color = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 3) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5) +
  labs(x = "Bandwidth", y = "AM risk", title = "Example 3") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",  # Display legend on top
    legend.key.size = unit(0.5, "lines")  # Set smaller legend key size
  ) +
  scale_color_discrete(
    name = "Model",
    breaks = labels,
    labels <- c("KDE", "KDE(VB)", "KDE-CV", "KDE(VB)-CV")
  )



##Combined plot of both exammples


library(ggplot2)

# Load first dataset
load("AM-EX2-bandwidthvary-check.RData")
AM_risk_KNN$example <- "Example 2"  
AM_risk_KNN_EX2 <- AM_risk_KNN      

# Load second dataset
load("AM-EX3-bandwidthvary-check.RData")
AM_risk_KNN$example <- "Example 3"
AM_risk_KNN_EX3 <- AM_risk_KNN

# Combine datasets
AM_risk_combined <- rbind(AM_risk_KNN_EX2, AM_risk_KNN_EX3)

# Set the label order
labels <- c("KDE", "KDE(VB)", "KDE-CV", "KDE(VB)-CV")

# Create linetype: dashed for "KDE" and "KDE-CV", solid otherwise
AM_risk_combined$linetype <- ifelse(AM_risk_combined$d %in% c("KDE", "KDE-CV"), "dashed", "solid")

# Create a flag for points: only show for KDE(SB) and KDE(SB)-CV
AM_risk_combined$show_point <- AM_risk_combined$d %in% c("KDE(VB)", "KDE(VB)-CV")

# Start plot
p <- ggplot(AM_risk_combined, aes(x = method, y = x, group = d, color = d, linetype = linetype)) +
  # Draw lines for everything
  geom_line(position = position_dodge(width = 0.1), size = 0.7) +
  # Draw crosses only for KDE(SB) and KDE(SB)-CV
  geom_point(
    data = subset(AM_risk_combined, show_point == TRUE),
    position = position_dodge(width = 0.1),
    size = 3,
    shape = 4  # shape 3 = cross ("+")
  ) +
  labs(x = "Bandwidth", y = "AM risk") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    legend.key.size = unit(0.5, "lines")
  ) +
  scale_color_discrete(
    name = "Model",
    breaks = labels,
    labels = labels
  ) +
  scale_linetype_manual(values = c("solid" = "solid", "dashed" = "dashed"), guide = "none") +
  facet_wrap(~ example)
p
# Save plot
ggsave("KDE-bandwidth.pdf", plot = p, width = 6, height = 4, dpi = 300)






library(ggplot2)
library(patchwork)

# Load first dataset
load("AM-EX2-bandwithvary-check.RData")
AM_risk_KNN$example <- "Example 2"  
AM_risk_KNN_EX2 <- AM_risk_KNN   
AM_risk_KNN_EX2$method <- gsub("H", "H0", AM_risk_KNN_EX2$method)

AM_risk_KNN_EX2$d <- as.character(AM_risk_KNN_EX2$d)
AM_risk_KNN_EX2$d[AM_risk_KNN_EX2$d == "KDE"] <- "KDE(L)"
AM_risk_KNN_EX2$d[AM_risk_KNN_EX2$d == "KDE(VB)"] <- "KDE(H)"
AM_risk_KNN_EX2$d[AM_risk_KNN_EX2$d == "KDE-CV"] <- "KDE(L)-CV"
AM_risk_KNN_EX2$d[AM_risk_KNN_EX2$d == "KDE(VB)-CV"] <- "KDE(H)-CV"

# Load second dataset
load("AM-EX3-bandwithvary-check.RData")
AM_risk_KNN$example <- "Example 3"
AM_risk_KNN_EX3 <- AM_risk_KNN
AM_risk_KNN_EX3$method <- gsub("H", "H0", AM_risk_KNN_EX3$method)

AM_risk_KNN_EX3$d <- as.character(AM_risk_KNN_EX3$d)
AM_risk_KNN_EX3$d[AM_risk_KNN_EX3$d == "KDE"] <- "KDE(L)"
AM_risk_KNN_EX3$d[AM_risk_KNN_EX3$d == "KDE(VB)"] <- "KDE(H)"
AM_risk_KNN_EX3$d[AM_risk_KNN_EX3$d == "KDE-CV"] <- "KDE(L)-CV"
AM_risk_KNN_EX3$d[AM_risk_KNN_EX3$d == "KDE(VB)-CV"] <- "KDE(H)-CV"

# Set the label order
labels <- c("KDE(L)", "KDE(H)", "KDE(L)-CV", "KDE(H)-CV")

# Set consistent colors
method_colors <- c(
  "KDE(L)" = "blue",      # Blue
  "KDE(H)" = "#2CA02C",      # Orange
  "KDE(L)-CV" = "red",   # Green
  "KDE(H)-CV" = "purple"    # Red
)

# Set factor levels for 'method'
AM_risk_KNN_EX2$method <- factor(AM_risk_KNN_EX2$method, levels = unique(AM_risk_KNN_EX2$method))
AM_risk_KNN_EX3$method <- factor(AM_risk_KNN_EX3$method, levels = unique(AM_risk_KNN_EX3$method))

# Function to add line type and point indicators
set_plot_attrs <- function(df) {
  df$linetype <- ifelse(df$d %in% c("KDE(L)", "KDE(L)-CV"), "dashed", "solid")
  df$show_point <- df$d %in% c("KDE(H)", "KDE(H)-CV")
  df
}

AM_risk_KNN_EX2 <- set_plot_attrs(AM_risk_KNN_EX2)
AM_risk_KNN_EX3 <- set_plot_attrs(AM_risk_KNN_EX3)

# Combine datasets
AM_risk_combined <- rbind(AM_risk_KNN_EX2, AM_risk_KNN_EX3)
AM_risk_combined <- set_plot_attrs(AM_risk_combined)

# Define the plotting function
make_plot <- function(data, title = NULL, ylab = NULL) {
  ggplot(data, aes(x = method, y = x, group = d, color = d, linetype = linetype)) +
    geom_line(position = position_dodge(width = 0.1), size = 0.7) +
    geom_point(
      data = subset(data, show_point == TRUE),
      position = position_dodge(width = 0.1),
      size = 3,
      shape = 4
    ) +
    labs(x = "H", y = ylab, title = title) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none",  # Remove legend from individual plots
      legend.key.size = unit(0.5, "lines"),
      strip.text = element_text(hjust = 0.5) # Center the facet labels
    ) +
    scale_color_manual(
      name = "Model",
      values = method_colors,
      breaks = labels,
      labels = labels
    ) +
    scale_linetype_manual(values = c("solid" = "solid", "dashed" = "dashed"), guide = "none")
}

# Create individual plots
plot_ex2 <- make_plot(AM_risk_KNN_EX2, "Example 2", "AM risk")
plot_ex3 <- make_plot(AM_risk_KNN_EX3, "Example 3", "AM risk")

# Modify plot for Example 3 to keep y-axis label
plot_ex3 <- plot_ex3 + theme(
  axis.title.y = element_text(text = "AM risk")
)

# Combine plots with shared legend
final_plot <- (plot_ex2 + plot_ex3) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.box.spacing = unit(0.5, "cm")
  )

# Show the final combined plot
print(final_plot)

# Save to PDF
ggsave("KDE-bandwidth1.pdf", plot = final_plot, width = 6, height = 3.5, dpi = 300)












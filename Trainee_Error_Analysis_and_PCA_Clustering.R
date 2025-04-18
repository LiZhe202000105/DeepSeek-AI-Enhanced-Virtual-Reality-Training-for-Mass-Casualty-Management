library(tidyverse)    
library(caret)        
library(pROC)         
library(stats)        
library(cluster)      
library(factoextra)   

data <- read.csv("S1_table1_data.csv")

error_rate <- apply(data[, -1], 2, function(x) mean(x == 0))
error_rate_df <- data.frame(Question = colnames(data[, -1]), ErrorRate = error_rate)
error_rate_df <- error_rate_df %>% arrange(desc(ErrorRate))

top_10_errors <- head(error_rate_df, 10)
print(top_10_errors)

data$TotalScore <- rowSums(data[, -1])  

data$Passed <- ifelse(data$TotalScore >= 50, 1, 0)

set.seed(123)
trainIndex <- createDataPartition(data$Passed, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

model <- glm(Passed ~ ., data = trainData[, -1], family = binomial)

predictions <- predict(model, testData, type = "response")
predicted_class <- ifelse(predictions > 0.5, 1, 0)

roc_curve <- roc(testData$Passed, predictions)
plot(roc_curve)
print(paste("AUC: ", auc(roc_curve)))

error_data <- data[, -c(1, ncol(data))]
error_data_binary <- ifelse(error_data == 0, 1, 0)

error_data_scaled <- scale(error_data_binary)

pca_result <- prcomp(error_data_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

pca_data <- data.frame(PC1 = pca_result$x[, 1], PC2 = pca_result$x[, 2], Class = data$Passed)

ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(Class))) +
  geom_point() +
  labs(title = "PCA-Based Clustering of Trainee Error Patterns",
       x = "Principal Component 1", y = "Principal Component 2")

set.seed(123)
kmeans_result <- kmeans(pca_data[, c("PC1", "PC2")], centers = 3)

fviz_cluster(kmeans_result, data = pca_data[, c("PC1", "PC2")], ellipse.type = "convex", geom = "point", ggtheme = theme_minimal())

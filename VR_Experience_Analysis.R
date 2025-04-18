library(randomForest)
library(caret)
library(ggplot2)
library(cluster)
library(corrplot)

S7_table4_data <- read.csv("path_to_your_file/S7_table4_data.csv")

head(S7_table4_data)

experience_factors <- S7_table4_data[, c("RealisticFeeling", "SimpleOperation", "EasyInterface", 
                                         "ConvenientEquipment", "MultiDimensionalViewing", "StimulatesInterest",
                                         "GoodLearningEffect", "ReplacesTraditionalTeaching", "NoDiscomfort", 
                                         "NoEyePain")]

experience_factors_scaled <- scale(experience_factors)

set.seed(123)
kmeans_result <- kmeans(experience_factors_scaled, centers = 3, nstart = 25)

S7_table4_data$Cluster <- as.factor(kmeans_result$cluster)

table(S7_table4_data$Cluster)

rf_model <- randomForest(TotalScore ~ RealisticFeeling + SimpleOperation + EasyInterface + 
                           ConvenientEquipment + MultiDimensionalViewing + StimulatesInterest +
                           GoodLearningEffect + ReplacesTraditionalTeaching + NoDiscomfort + 
                           NoEyePain, data = S7_table4_data, importance = TRUE)

importance(rf_model)

varImpPlot(rf_model)

linear_model <- lm(TotalScore ~ RealisticFeeling + SimpleOperation + EasyInterface + 
                    ConvenientEquipment + MultiDimensionalViewing + StimulatesInterest + 
                    GoodLearningEffect + ReplacesTraditionalTeaching + NoDiscomfort + 
                    NoEyePain, data = S7_table4_data)

summary(linear_model)

cor_matrix <- cor(experience_factors)

corrplot(cor_matrix, method = "circle", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

write.csv(importance(rf_model), "Variable_Importance_RF.csv")
write.csv(summary(linear_model)$coefficients, "Linear_Model_Summary.csv")

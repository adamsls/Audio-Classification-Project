#################################################
#Linda Adams - Freesound Audio Tagging Challenge
#################################################

#ML Model

#################################################
library(caret)
library(mlbench)

train_data <- read.csv('TRAINCSV.csv', stringsAsFactors = TRUE, header = TRUE)
test_data <- read.csv('TESTCSV.csv', header = TRUE, stringsAsFactors = TRUE)

#csv file for kaggle submission
real_test_data <- read.csv("Data/sample_submission.csv")

#remove labels from test data
test_no_labels <- test_data[1:41]

#remove id column from train_data
train_data <- train_data[-1]

#set parameters
set.seed(42)

tune_grid <- data.frame(mtry = c(1:30), splitrule = "extratrees",
                       min.node.size = 1)

model <- train(label ~., data = train_data, method = "ranger", trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

prediction_vector <- predict(model, test_no_labels)

real_test_data['label'] <- prediction_vector

write.csv(real_test_data, file = "My Submission.csv")

mean(prediction_vector == real_test_data['label'])

#Wifi Fingerprinting - Load Data
install.packages("caret")
setwd("~/Desktop/IOT Analytics/Wifi location")
library(readr)
trainingData <-read_csv("~/Desktop/IOT Analytics/Wifi location/UJIndoorLoc/trainingData.csv")
View(trainingData)
validationData<- read_csv("~/Desktop/IOT Analytics/Wifi location/UJIndoorLoc/validationData.csv")
View(validationData)

#Training Dataset 
Wifi_train <- trainingData
str(Wifi_train)
summary(Wifi_train)
getOption("max.print")
sapply(Wifi_train, mean, na.rm=TRUE)
head(Wifi_train)
attributes(Wifi_train)
is.na(Wifi_train)
length(Wifi_train)
anyNA(Wifi_train)

#Validation_test Dataset
wifi_valid <- validationData
str(wifi_valid)
summary(wifi_valid)
head(wifi_valid)

#Feature Engineering
wifi_valid$TIMESTAMP<- NULL
wifi_valid$LONGITUDE<-NULL
wifi_valid$LATITUDE<-NULL
wifi_valid$USERID<-NULL
wifi_valid$PHONEID<-NULL
View(wifi_valid)
head(wifi_valid)
str(wifi_valid)





#Multi-dv to include 523(floor), 524(buildingid), 525(spaceid), 526(relativeposition)
wifi_valid$Combined_dv <- as.factor(paste(wifi_valid$FLOOR, wifi_valid$SPACEID, wifi_valid$BUILDINGID, wifi_valid$RELATIVEPOSITION, sep = ""))
View(wifi_valid)
attributes(wifi_valid)



#Subset data for sampling 

wifi_sample <- wifi_valid[sample(1:nrow(wifi_valid), 500, replace = FALSE),]
View(wifi_sample)
str(wifi_sample)

#Feature Engineering for trianing data set

training_main_sample <-wifi_sample[-c(1,4,92:95,158:160,215:221,226:227,238:247,291:293,296:309,333,347:365,406:451,457:477,482:488,491,497,518:520)]
View(training_main_sample)
attributes(training_main_sample)

training_main_sample$FLOOR<-NULL
training_main_sample$BUILDINGID<-NULL
training_main_sample$SPACEID<-NULL
training_main_sample$RELATIVEPOSITION<-NULL
str(training_main_sample)
attributes(training_main_sample)


#Split data set LineaR Model_WIFI_75% split
library(caret)
set.seed(123)
inTrain_WIFI <- createDataPartition(training_main_sample$Combined_dv, p = .75, list = FALSE)
inTrain_WIFI
summary(inTrain_WIFI)
attributes(inTrain_WIFI)
str(inTrain_WIFI)
View(inTrain_WIFI)

#Training the models 70/30 split_WIFI

training_type_WIFI <- training_main_sample[inTrain_WIFI,]
testing_type_WIFI <- training_main_sample[-inTrain_WIFI,]

training_type_WIFI
testing_type_WIFI

attributes(training_type_WIFI)
attributes(testing_type_WIFI)

#10 fold cross validation_WIFI
fitControl_type_WIFI <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
fitControl_type_WIFI

#Linear Model_Wifi
set.seed(123)
LMfIT_TYPE_wifi <- train(Combined_dv~., data = training_type_WIFI, method ="lm", trControl = fitControl_type_WIFI)
LMfIT_TYPE_wifi 
summary(LMfIT_TYPE)

#SVM_WIFI for sample
set.seed(123)
anyNA(training_type_WIFI)
SVMFit_wifi <- train(Combined_dv~., data = training_type_WIFI, method = "svmLinear", trControl=fitControl_type_WIFI, preProcess =c("center", "scale"), tuneLength = 10)
SVMFit_wifi
summary(SVMFit_wifi)
predictors(SVMFit_wifi)
attributes(SVMFit_wifi)

#Predict SVM for Wifi
set.seed(123)
SVMPredict_wifi <- predict(SVMFit_wifi, testing_type_WIFI)
SVMPredict_wifi

summary(SVMPredict_wifi)
postResample(SVMPredict_wifi, testing_type_WIFI$Combined_dv)
confusionMatrix(SVMPredict_wifi, testing_type_WIFI$Combined_dv)

#Tune SVM Model_GRID_optimized for SVM_Grid
SVM_grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(123)
svm_Linear_Grid <- train(Combined_dv ~ ., data = training_type_WIFI, method = "svmLinear", trControl=fitControl_type_WIFI,preProcess = c("center", "scale"),tuneGrid = SVM_grid,tuneLength = 10)

svm_Linear_Grid
plot(svm_Linear_Grid)


test_svm_grid <- predict(svm_Linear_Grid, testing_type_WIFI)
test_svm_grid
postResample(test_svm_grid, testing_type_WIFI$Combined_dv)


#Random Forrest_Wifi
library(randomForest)
library(caret)
install.packages("MultivariateRandomForest")
library(MultivariateRandomForest)
set.seed(123)

#RandomForrest_100
WifiForrest100 <- train(Combined_dv~.,  data = training_type_WIFI, method = "rf", metric = "mse", ntree = 100)
WifiForrest100
summary(WifiForrest100)
predictors(TypeForrest100)

#RandomForrest_189_best model 
WifiForrest189 <- train(Combined_dv~.,  data = training_type_WIFI, method = "rf", metric = "mse", ntree = 189)
WifiForrest189
summary(WifiForrest189)
predictors(WifiForrest189)
confusionMatrix(WifiForrest189)

#prediction RF 189 TREES
PredictRF_wifi <- predict(WifiForrest189, testing_type_WIFI)
PredictRF_wifi
postResample(PredictRF_wifi, testing_type_WIFI$Combined_dv)

#KNN wifi
set.seed(123)
install.packages("class")
library(class)
library(caret)

KNN_wifi <- train(Combined_dv ~ ., data = training_type_WIFI, method = "knn", trControl=fitControl_type_WIFI, tuneLength=20,  preProcess =c("center", "scale"))
KNN_wifi
summary(KNN_wifi)
predictors(KNN_wifi)
attributes(KNN_wifi)

#KNN prediction_wifi
KNNpred_wifi <- predict(KNN_wifi, testing_type_WIFI)
KNNpred_wifi
postResample(KNNpred_wifi, testing_type_WIFI$Combined_dv)




#c50
install.packages("C50")
library(C50)
set.seed(123)
str(wifi_valid)

g<- runif(nrow(training_main_sample))
wifi_random <- training_main_sample[order(g),]
str(wifi_random)
View(wifi_random)
attributes(wifi_random)
wifi_random$Combined_dv <- as.factor(wifi_random$Combined_dv)
str((wifi_random$Combined_dv))

C50_MODEL1 <- C5.0(wifi_random[-377], wifi_random$Combined_dv)
C50_MODEL1
summary(C50_MODEL1)
C50_predict <- predict(C50_MODEL1, testing_type_WIFI)
C50_predict
postResample(C50_predict, testing_type_WIFI$Combined_dv)





  
  
  









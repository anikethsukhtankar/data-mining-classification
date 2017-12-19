#Function to Read File
readfile <- function(path_data){
  require(readxl)
  return (read_excel(path_data));
}

#Function to normalize all data values
normalize <- function(x) {
  norm <- ((x - min(x))/(max(x) - min(x)))
  return (norm)
}

#Function to calculate Accuracy, Precision, Recall and fMeasure Values
calculateStats <- function(cf) {
  accuracy <- cf$overall['Accuracy']
  compTable <- as.table(cf)
  
  precision = numeric() 
  for (i in 1:ncol(compTable)) {
    precision <- c(precision,compTable[i,i]/sum(compTable[i,]))
  }
  
  recall = numeric() 
  for (i in 1:ncol(compTable)) {
    recall <- c(recall,compTable[i,i]/sum(compTable[,i]))
  }
  
  #f Measure = 2rp / r+p
  
  fMeasure = (2 * recall * precision)/(recall+precision)
  
  return(list(accuracy,precision,recall,fMeasure))
}

#Function for Data Preprocessing
divideDataset <- function(proj1){
  countryColNum <- grep("Country",names(proj1))
  proj1 <- proj1[,-countryColNum]
  proj1_n <- as.data.frame(lapply(proj1[,-5],normalize))
  proj1_n$Continent <- proj1$Continent
  proj1_n[,"train"] <- ifelse(runif(nrow(proj1_n))<0.8,1,0)
  
  trainset <- proj1_n[proj1_n$train==1,]
  testset <- proj1_n[proj1_n$train==0,]
  
  trainColNum <- grep("train",names(trainset))
  trainset <- trainset[,-trainColNum]
  testset <- testset[,-trainColNum]
  
  trainset$Continent <- factor(trainset$Continent)
  testset$Continent <- factor(testset$Continent)
  
  return(list(trainset,testset));
}

#Learns a fit based on given training set using the Support Vector Machine Algorithm. Returns a fit.
mySVM <- function(trainset){
  svm_model <- svm(Continent~ ., data=trainset, method="C-classification", kernel="radial", type = "C")
  obj <- tune.svm(Continent~., data = trainset, gamma = 2^(-1:1), cost = 2^(2:4))
  svm_model_after_tune <- svm(Continent~ ., data=trainset, method="C-classification", kernel="radial", type = "C", cost=obj$best.parameters$cost, obj$best.parameters$gamma)
  return (svm_model_after_tune)
}

#For each sample in the test set it uses the svm model to predict the label returns a confusion matrix
mySVMPredict <- function(svm_model,testset){
  pred_test <- predict(svm_model,testset)
  cf <- confusionMatrix(pred_test, testset$Continent)
  return (cf);
}

#Learns a fit based on given training set using the K Nearest Neighbor Algorithm. Returns a fit.
myKNN <- function(trainset){
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  return (train(as.data.frame(trainset[, 1:4]),make.names(trainset$Continent), method='knn', trControl=trctrl, preProcess = c("center", "scale"), tuneLength = 10))
}

#For each sample in the test set it uses the knn model to predict the label returns a confusion matrix
myKNNPredict <- function(knn_model,testset){
  pred_test <- predict(knn_model,as.data.frame(testset[, 1:4]))
  cf <- confusionMatrix(pred_test, make.names(testset$Continent))
  return (cf);
}

#Learns a fit based on given training set using the RIPPER Algorithm. Returns a fit.
myRIPPER <- function(trainset){
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  return (train(as.data.frame(trainset[, 1:4]),make.names(trainset$Continent), method='JRip', trControl=trctrl, preProcess = c("center", "scale"), tuneLength = 5))
}

#For each sample in the test set it uses the RIPPER model to predict the label returns a confusion matrix
myRIPPERPredict <- function(rip_model,testset){
  pred_test <- predict(rip_model,as.data.frame(testset[, 1:4]))
  cf <- confusionMatrix(pred_test, make.names(testset$Continent))
  return (cf);
}

#Learns a fit based on given training set using the C4.5 Algorithm. Returns a fit.
myC45 <- function(trainset){
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  return (train(as.data.frame(trainset[, 1:4]),make.names(trainset$Continent), method='J48', trControl=trctrl, preProcess = c("center", "scale"), tuneLength = 5))
}

#For each sample in the test set it uses the C4.5 model to predict the label returns a confusion matrix
myC45Predict <- function(c45_model,testset){
  pred_test <- predict(c45_model,as.data.frame(testset[, 1:4]))
  cf <- confusionMatrix(pred_test, make.names(testset$Continent))
  return (cf);
}

Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre7')
#install.packages(c("rJava","RWeka","class","zeallot","gmodels","caret","e1071"))
require(rJava)
require(RWeka)
require(class)
require(zeallot)
require(gmodels)
require(caret)
require(e1071)

set.seed(1707) #SET SEED VALUES HERE

proj1 <-readfile("DATASET_FILEPATH HERE") #SET DATASET_FILEPATH HERE TO YOUR PATH
c(trainset,testset)%<-%divideDataset(proj1)

#Implementation for Support Vector Machine
svm_model <- mySVM(trainset)
svm_model
cf <- mySVMPredict(svm_model,testset)
c(accuracy,precision,recall,fMeasure)%<-%calculateStats(cf)
accuracy
precision
recall
fMeasure
cf

#Implementation for K Nearest Neighbors
knn_model <- myKNN(trainset)
knn_model
cf1 <- myKNNPredict(knn_model,testset)
c(accuracy,precision,recall,fMeasure)%<-%calculateStats(cf1)
accuracy
precision
recall
fMeasure
cf1

#Implementation for RIPPER
rip_model <- myRIPPER(trainset)
rip_model
cf2 <- myRIPPERPredict(rip_model,testset)
c(accuracy,precision,recall,fMeasure)%<-%calculateStats(cf2)
accuracy
precision
recall
fMeasure
cf2

#Implementation for C4.5
c45_model <- myC45(trainset)
c45_model
cf3 <- myC45Predict(c45_model,testset)
c(accuracy,precision,recall,fMeasure)%<-%calculateStats(cf3)
accuracy
precision
recall
fMeasure
cf3

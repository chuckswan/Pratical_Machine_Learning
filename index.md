---
title: "Practical Machine Learning Coursera Peer Assessment"
author: "Charles Swanson"
date: "December 17, 2018"
output: 
  html_document: 
    keep_md: yes
---



## Summary

This report will use a machine learning algorithm to predict the manner (correctly or incorrectly) in which users performed dumbell lifts. 

The report will describe how I built the prediction model, tesing cross validation, the expected out of sample error, and why I made the choices I did. I will also use the prediction model to predict 20 different test cases.


### Project Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here:](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

### Training and Test Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Set environment


```r
library(knitr)
opts_chunk$set(echo = TRUE, cache= TRUE, results = 'hold')
```

### Load libraries 


```r
library(caret)
library(rpart)
library(randomForest)
library(RCurl)
setwd("~/00 Work Documents/Coursera/Machine_Learning")
```

Load Data


```r
trainingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
Training_Data  <- read.csv(text = trainingLink, header=TRUE, sep=",", na.strings=c("NA",""))
Training_Data  <- Training_Data[,-1] # Removed the first column as it contains an ID
```

### Data Sets Partitions Definitions

Create data partitions for training and validation data sets.


```r
inTrain = createDataPartition(Training_Data$classe, p=0.60, list=FALSE)
training = Training_Data[inTrain,]
validating = Training_Data[-inTrain,]
```
## Data Exploration and Cleaning

After looking at the data, there are many columns missing considerable data. I chose to remove columns with less than 80 percent of the data. 


```r
Retain <- c((colSums(!is.na(training[,-ncol(training)])) >= 0.8*nrow(training)))
training   <-  training[,Retain]
validating <- validating[,Retain]
```

## Modeling

I am chosing to start with Random Forest model since in random forests.


```r
model <- randomForest(classe~.,data=training)
print(model)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.14%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    1    0    0    0 0.0002986858
## B    2 2277    0    0    0 0.0008775779
## C    0    2 2048    4    0 0.0029211295
## D    0    0    5 1924    1 0.0031088083
## E    0    0    0    2 2163 0.0009237875
```


Cross validation of the model using validation data with confusion Matrix.


```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    0    0    0
##          B    1 1518    3    0    0
##          C    0    0 1364    1    0
##          D    0    0    1 1285    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9992          
##                  95% CI : (0.9983, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.999           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   1.0000   0.9971   0.9992   1.0000
## Specificity            1.0000   0.9994   0.9998   0.9998   1.0000
## Pos Pred Value         1.0000   0.9974   0.9993   0.9992   1.0000
## Neg Pred Value         0.9998   1.0000   0.9994   0.9998   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1738   0.1638   0.1838
## Detection Prevalence   0.2843   0.1940   0.1740   0.1639   0.1838
## Balanced Accuracy      0.9998   0.9997   0.9985   0.9995   1.0000
```


### Expected out-of-sample error
The expected out-of-sample error is estimated at 0.001, or 0.1%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set.


### Predicting New Values with Testing Dataset

We will predict the Class values from the testing csv file provided. We will process the data as we did previously. 

#### Getting Testing Dataset


```r
testingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
# Apply the Same transformations as done to the training dataset
Testing_Data  <- read.csv(text = testingLink, header=TRUE, sep=",", na.strings=c("NA",""))
Testing_Data <- Testing_Data[,-1] # Remove the first column that represents a ID Row
Testing_Data <- Testing_Data[, Retain] # Keep the same columns as the training dataset
Testing_Data <- Testing_Data[,-ncol(Testing_Data)] # Remove the problem ID

testing_dataset <- rbind(training[100, -59] , Testing_Data) 

# Apply the ID Row to row.names and 100 for dummy row from testing dataset 
row.names(testing_dataset) <- c(100, 1:20)
```

#### Predicting with testing dataset


```r
predictions <- predict(model,newdata=testing_dataset[-1,])
print(predictions)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

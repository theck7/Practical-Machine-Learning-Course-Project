---
title: "Practical Machine Learning Course Project"
output: 
  html_document:
    keep_md: true
---


```r
#Load required packages and set the seed for reproducibility
library(dplyr)
library(ggplot2)
library(caret)
library(parallel)
library(doParallel)
library(rattle)

set.seed(12563)
```

# Executive Summary

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did an exercise. They were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:

1. Class A: exactly according to the specification
2. Class B: throwing the elbows to the front
3. Class C lifting the dumbbell only halfway
4. Class D lowering the dumbbell only halfway
5. Class E throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har

#Load Data & Exploratory Data Analysis

```r
#Load the data from working directory
training <- read.csv("pml-training.csv", header = TRUE, sep = ",", na.strings=c("NA","#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", header = TRUE, sep = ",", na.strings=c("NA","#DIV/0!", ""))
```




```r
dim(training)
```

```
## [1] 19622   160
```

```r
str(training)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

The training data set has 160 variables, many which consist primarily of missing data.


```r
summarize(group_by(training, classe), Training=n())
```

```
## # A tibble: 5 x 2
##   classe Training
##   <fct>     <int>
## 1 A          5580
## 2 B          3797
## 3 C          3422
## 4 D          3216
## 5 E          3607
```

```r
qplot(training$classe, fill=I("green"), col=I("black"), main = "Number of Observations per Class", xlab="Class", ylab="Frequency")
```

![](index_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

As can be seen from the table and chart above, each class has a large amount of observations with Class A, performing the exercise exactly according to the specification, having the most data.  

#Clean Data

```r
#Delete columns with missing values
training <- training[ , colSums(is.na(training)) == 0]
testing <- testing[ , colSums(is.na(testing)) == 0]

#Delete irrelevant variables (first 7 columns)
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

The first 7 columns contain user and time information which will not be useful variables for the model.  These 7 variables, along with 100 variables that consist almost entirely of missing values are removed from the data set.  This leaves the training and test data sets with 53 variables which will be used in a model to predict the manner in which an exercise was done.

#Cross Validation
Cross-validation will be performed by subsampling the training data set randomly without replacement into 2 data sets: 75% of training data set for training and 25% for testing. The models will be fitted on the subtraining data set, and tested on the subtesting data set. 

The test set accuracy will be estimated with the training set. Once the most accurate model is found, it will be tested on the original testing data set.

Summary of Approach:

1. Use the training set
2. Split it into training/test sets
3. Build a model on the training set
4. Evaluate on the test set


```r
#Partition training set
Data_Partition <- createDataPartition(y = training$classe, p = 0.75, list = FALSE)
subtraining <- training[Data_Partition, ]
subtesting <- training[-Data_Partition, ]
```

#Decision Tree Model
The first model to be evaluated is a decision tree.

```r
#Decision Tree
mod_DT <- train(classe ~ ., method= "rpart", data = subtraining)

#Plot the Decision Tree
fancyRpartPlot(mod_DT$finalModel, sub = "")
```

![](index_files/figure-html/Decision_Tree_Model-1.png)<!-- -->


```r
#Test results on subtesting data set
prediction_DT <- predict(mod_DT, subtesting)
DT_CM <- confusionMatrix(prediction_DT, subtesting$classe)

DT_Accuracy <- DT_CM$overall['Accuracy']

DT_CM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1266  381  396  358  134
##          B   20  345   24  151  128
##          C  103  223  435  295  242
##          D    0    0    0    0    0
##          E    6    0    0    0  397
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4982          
##                  95% CI : (0.4841, 0.5123)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3443          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9075  0.36354   0.5088   0.0000  0.44062
## Specificity            0.6384  0.91833   0.7869   1.0000  0.99850
## Pos Pred Value         0.4994  0.51647   0.3351      NaN  0.98511
## Neg Pred Value         0.9455  0.85741   0.8835   0.8361  0.88802
## Prevalence             0.2845  0.19352   0.1743   0.1639  0.18373
## Detection Rate         0.2582  0.07035   0.0887   0.0000  0.08095
## Detection Prevalence   0.5169  0.13622   0.2647   0.0000  0.08218
## Balanced Accuracy      0.7729  0.64094   0.6478   0.5000  0.71956
```

The accuracy of the decision tree model is only 0.4981648.

#Random Forest Model
The next model to be evaluated is a random forest model which has higher accuracy at the expense of speed, interpretability, and a greater risk of overfitting.

```r
#Random Forest Model
#Configure parallel processing
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#Configure trainControl object
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

#Develop training model
mod_RF <- train(classe ~ ., method = "rf", data = subtraining, verbose = FALSE,trControl = fitControl)

#De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

#Prediction
prediction_RF <- predict(mod_RF, subtesting)
RF_CM <- confusionMatrix(prediction_RF, subtesting$classe)

RF_Accuracy <- RF_CM$overall['Accuracy']
RF_CI <- RF_CM$overall[3:4]

RF_CM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1388    8    0    0    0
##          B    6  938    3    0    0
##          C    0    3  847   11    1
##          D    0    0    5  793    1
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9891, 0.9943)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9899          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9950   0.9884   0.9906   0.9863   0.9978
## Specificity            0.9977   0.9977   0.9963   0.9985   0.9998
## Pos Pred Value         0.9943   0.9905   0.9826   0.9925   0.9989
## Neg Pred Value         0.9980   0.9972   0.9980   0.9973   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2830   0.1913   0.1727   0.1617   0.1833
## Detection Prevalence   0.2847   0.1931   0.1758   0.1629   0.1835
## Balanced Accuracy      0.9964   0.9931   0.9935   0.9924   0.9988
```

The accuracy of the random forest model is 0.9920473 with a 95% confidence interval of (0.9891443, 0.9943389).  This is much better than the decision tree and thus the random forest model is chosen.  

Out of Sample Error is the error rate you get on a new data set. Sometimes called generalization error. Out of sample error is what you care about. The expected out of sample error will equal 1-accuracy in the cross-validation data which is 0.0079527

#Prediction Test Cases
Below are the results of the random forest model applied to the testing dataset:


```r
#Prediction Quiz 
Prediction_Quiz <- predict(mod_RF, testing)
Prediction_Quiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

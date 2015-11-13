# Lecture 2
Alexander Tuzhikov  
November 12, 2015  

#Caret functionality

* Some preprocessing (cleaning)
    * `preProcess`
* Data splitting
    * `createDataPartition`
    * `createResample`
    * `createTimeSlices`
* Training/testing funcitions
    * `train`
    * `predict`
* Model comparison
    * `confusionMatrix`
    
`caret` allows to unify the parameters for a wide variaty of predicting models.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)
```

```
## [1] 3451   58
```

```r
set.seed(32343)
modelFit <- train(type ~., data = training, method="glm")
modelFit$finalModel
```

```
## 
## Call:  NULL
## 
## Coefficients:
##       (Intercept)               make            address  
##        -1.672e+00         -4.711e-01         -1.647e-01  
##               all              num3d                our  
##         8.728e-02          2.787e+00          7.045e-01  
##              over             remove           internet  
##         8.066e-01          2.230e+00          7.391e-01  
##             order               mail            receive  
##         7.448e-01          1.344e-01         -1.031e-01  
##              will             people             report  
##        -1.144e-01         -7.938e-02          1.181e-01  
##         addresses               free           business  
##         1.007e+00          1.080e+00          1.097e+00  
##             email                you             credit  
##         1.012e-01          9.281e-02          7.377e-01  
##              your               font             num000  
##         2.068e-01          1.764e-01          1.914e+00  
##             money                 hp                hpl  
##         2.786e-01         -2.167e+00         -2.091e+00  
##            george             num650                lab  
##        -2.052e+01          5.929e-01         -4.047e+00  
##              labs             telnet             num857  
##        -2.682e-02         -1.339e-01          9.634e+00  
##              data             num415              num85  
##        -4.531e-01          9.340e-01         -1.821e+00  
##        technology            num1999              parts  
##         1.417e+00          3.529e-01         -5.781e-01  
##                pm             direct                 cs  
##        -9.872e-01         -3.670e-01         -4.180e+01  
##           meeting           original            project  
##        -2.677e+00         -7.488e-01         -1.626e+00  
##                re                edu              table  
##        -8.088e-01         -1.599e+00         -2.444e+00  
##        conference      charSemicolon   charRoundbracket  
##        -4.398e+00         -1.268e+00         -6.114e-01  
## charSquarebracket    charExclamation         charDollar  
##        -5.276e-01          3.308e-01          5.245e+00  
##          charHash         capitalAve        capitalLong  
##         2.634e+00          4.494e-02          1.064e-02  
##      capitalTotal  
##         7.528e-04  
## 
## Degrees of Freedom: 3450 Total (i.e. Null);  3393 Residual
## Null Deviance:	    4628 
## Residual Deviance: 1292 	AIC: 1408
```

```r
predictions <- predict(modelFit, newdata=testing)
predictions
```

```
##    [1] spam    spam    spam    spam    spam    spam    nonspam spam   
##    [9] spam    spam    spam    spam    spam    spam    spam    spam   
##   [17] spam    spam    spam    spam    spam    spam    spam    spam   
##   [25] spam    spam    spam    spam    spam    spam    spam    spam   
##   [33] spam    spam    spam    spam    spam    spam    spam    spam   
##   [41] nonspam nonspam spam    spam    spam    spam    spam    spam   
##   [49] nonspam spam    spam    spam    nonspam spam    spam    spam   
##   [57] spam    spam    spam    spam    spam    spam    spam    spam   
##   [65] spam    spam    nonspam spam    spam    nonspam spam    spam   
##   [73] spam    spam    spam    nonspam spam    spam    spam    spam   
##   [81] spam    spam    spam    spam    spam    spam    spam    spam   
##   [89] spam    nonspam spam    spam    spam    spam    spam    spam   
##   [97] spam    spam    spam    spam    spam    spam    spam    spam   
##  [105] spam    spam    spam    spam    spam    spam    spam    spam   
##  [113] nonspam spam    spam    spam    nonspam spam    spam    spam   
##  [121] nonspam spam    spam    spam    nonspam spam    spam    spam   
##  [129] spam    spam    spam    spam    spam    spam    nonspam nonspam
##  [137] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [145] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [153] spam    spam    spam    nonspam spam    nonspam spam    spam   
##  [161] spam    spam    spam    spam    spam    spam    spam    spam   
##  [169] nonspam spam    spam    spam    spam    spam    spam    spam   
##  [177] spam    spam    spam    spam    spam    spam    spam    spam   
##  [185] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [193] spam    spam    spam    spam    spam    spam    spam    spam   
##  [201] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [209] spam    spam    spam    spam    spam    spam    spam    spam   
##  [217] spam    nonspam spam    spam    nonspam spam    spam    spam   
##  [225] spam    spam    spam    spam    spam    spam    spam    spam   
##  [233] spam    spam    spam    spam    spam    spam    spam    spam   
##  [241] spam    nonspam nonspam spam    spam    spam    spam    spam   
##  [249] nonspam spam    spam    spam    spam    spam    spam    spam   
##  [257] spam    spam    spam    spam    spam    nonspam nonspam spam   
##  [265] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [273] spam    spam    spam    spam    spam    spam    spam    spam   
##  [281] spam    nonspam spam    spam    spam    spam    spam    spam   
##  [289] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [297] spam    nonspam spam    spam    spam    spam    spam    spam   
##  [305] spam    spam    spam    spam    spam    spam    spam    spam   
##  [313] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [321] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [329] spam    spam    nonspam nonspam spam    spam    spam    spam   
##  [337] spam    spam    spam    spam    spam    spam    spam    spam   
##  [345] spam    spam    spam    spam    spam    spam    spam    spam   
##  [353] spam    nonspam spam    spam    spam    spam    spam    spam   
##  [361] spam    spam    spam    spam    spam    spam    spam    spam   
##  [369] nonspam spam    spam    nonspam spam    spam    spam    spam   
##  [377] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [385] spam    spam    nonspam spam    spam    spam    nonspam spam   
##  [393] nonspam spam    spam    nonspam nonspam spam    nonspam nonspam
##  [401] spam    spam    spam    spam    spam    spam    spam    spam   
##  [409] spam    spam    spam    nonspam spam    nonspam nonspam spam   
##  [417] nonspam nonspam spam    spam    nonspam nonspam nonspam spam   
##  [425] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [433] spam    spam    spam    spam    spam    spam    spam    spam   
##  [441] spam    spam    spam    spam    spam    spam    spam    spam   
##  [449] spam    spam    spam    spam    spam    nonspam nonspam nonspam
##  [457] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [465] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [473] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [481] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [489] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [497] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [505] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [513] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [521] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [529] nonspam spam    nonspam nonspam spam    nonspam nonspam nonspam
##  [537] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [545] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [553] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [561] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [569] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [577] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [585] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [593] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [601] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [609] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [617] nonspam spam    nonspam nonspam spam    nonspam nonspam nonspam
##  [625] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [633] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [641] nonspam spam    nonspam nonspam spam    nonspam nonspam spam   
##  [649] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [657] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [665] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [673] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [681] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [689] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [697] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [705] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [713] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [721] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [729] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [737] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [745] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
##  [753] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [761] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [769] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [777] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [785] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [793] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
##  [801] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [809] nonspam nonspam nonspam nonspam nonspam nonspam spam    spam   
##  [817] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [825] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [833] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [841] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [849] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [857] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [865] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [873] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [881] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
##  [889] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [897] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [905] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [913] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [921] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [929] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [937] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [945] nonspam spam    nonspam nonspam nonspam nonspam nonspam spam   
##  [953] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [961] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [969] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [977] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [985] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [993] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1001] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
## [1009] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1017] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
## [1025] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1033] nonspam spam    nonspam nonspam nonspam spam    nonspam nonspam
## [1041] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1049] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1057] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
## [1065] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1073] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1081] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1089] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
## [1097] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1105] spam    nonspam spam    nonspam nonspam spam    nonspam nonspam
## [1113] spam    nonspam nonspam nonspam nonspam nonspam spam    nonspam
## [1121] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1129] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1137] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1145] nonspam nonspam nonspam nonspam nonspam nonspam
## Levels: nonspam spam
```

```r
confusionMatrix(predictions, testing$type)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction nonspam spam
##    nonspam     661   57
##    spam         36  396
##                                           
##                Accuracy : 0.9191          
##                  95% CI : (0.9018, 0.9342)
##     No Information Rate : 0.6061          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.8293          
##  Mcnemar's Test P-Value : 0.03809         
##                                           
##             Sensitivity : 0.9484          
##             Specificity : 0.8742          
##          Pos Pred Value : 0.9206          
##          Neg Pred Value : 0.9167          
##              Prevalence : 0.6061          
##          Detection Rate : 0.5748          
##    Detection Prevalence : 0.6243          
##       Balanced Accuracy : 0.9113          
##                                           
##        'Positive' Class : nonspam         
## 
```

#Data slicing


```r
set.seed(32323)
folds<- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
sapply(folds, length)
```

```
## Fold01 Fold02 Fold03 Fold04 Fold05 Fold06 Fold07 Fold08 Fold09 Fold10 
##   4141   4140   4141   4142   4140   4142   4141   4141   4140   4141
```

```r
folds[[1]][1:10]
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10
```

```r
set.seed(32323)
folds<- createResample(y=spam$type, times=10, list=TRUE)
sapply(folds, length)
```

```
## Resample01 Resample02 Resample03 Resample04 Resample05 Resample06 
##       4601       4601       4601       4601       4601       4601 
## Resample07 Resample08 Resample09 Resample10 
##       4601       4601       4601       4601
```

```r
folds[[1]][1:10]
```

```
##  [1]  1  2  3  3  3  5  5  7  8 12
```

```r
set.seed(32323)
tme<- 1:1e3
folds<- createTimeSlices(y=tme, initialWindow = 20, horizon = 10) #horizon is the number of points you are going to be predicting
names(folds)
```

```
## [1] "train" "test"
```

```r
folds$train[[1]]
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
```

```r
folds$test[[1]]
```

```
##  [1] 21 22 23 24 25 26 27 28 29 30
```

#Training options


```r
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)
```

```
## [1] 3451   58
```

```r
set.seed(32343)
modelFit <- train(type ~., data = training, method="glm")
args(train.default)
```

```
## function (x, y, method = "rf", preProcess = NULL, ..., weights = NULL, 
##     metric = ifelse(is.factor(y), "Accuracy", "RMSE"), maximize = ifelse(metric %in% 
##         c("RMSE", "logLoss"), FALSE, TRUE), trControl = trainControl(), 
##     tuneGrid = NULL, tuneLength = 3) 
## NULL
```

```r
#needs trainControls()
```

##Continous outcomes:  

* RMSE = Root Mean Squared Error
* RSquared = R^2 from regression models

##Categorical outcomes:  

* Accurcy - Fraction correct
* Kappa - A measure of concordance

##`trainControl` resampling

* `method`

    * boot = bootstrapping
    * boot632 = bootstrapping with adjustment
    * cv = cross validation
    * repeatedcv = repeated cross validation
    * LOOCV = leave one out cross validation

* `number`
    
    * For boot/cross validation
    * Number of subsamples to take
    
* `repeats`

    * Number of times to repeate subsampling
    * if big this can slow things down

###Setting the seed

* it is often useful to set an overall seed
* you can also set a seed for each resample
* seeding each resample is useful for parallel fits

#Plotting predictors


```r
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
summary(Wage)
```

```
##       year           age               sex                    maritl    
##  Min.   :2003   Min.   :18.00   1. Male  :3000   1. Never Married: 648  
##  1st Qu.:2004   1st Qu.:33.75   2. Female:   0   2. Married      :2074  
##  Median :2006   Median :42.00                    3. Widowed      :  19  
##  Mean   :2006   Mean   :42.41                    4. Divorced     : 204  
##  3rd Qu.:2008   3rd Qu.:51.00                    5. Separated    :  55  
##  Max.   :2009   Max.   :80.00                                           
##                                                                         
##        race                   education                     region    
##  1. White:2480   1. < HS Grad      :268   2. Middle Atlantic   :3000  
##  2. Black: 293   2. HS Grad        :971   1. New England       :   0  
##  3. Asian: 190   3. Some College   :650   3. East North Central:   0  
##  4. Other:  37   4. College Grad   :685   4. West North Central:   0  
##                  5. Advanced Degree:426   5. South Atlantic    :   0  
##                                           6. East South Central:   0  
##                                           (Other)              :   0  
##            jobclass               health      health_ins      logwage     
##  1. Industrial :1544   1. <=Good     : 858   1. Yes:2083   Min.   :3.000  
##  2. Information:1456   2. >=Very Good:2142   2. No : 917   1st Qu.:4.447  
##                                                            Median :4.653  
##                                                            Mean   :4.654  
##                                                            3rd Qu.:4.857  
##                                                            Max.   :5.763  
##                                                                           
##       wage       
##  Min.   : 20.09  
##  1st Qu.: 85.38  
##  Median :104.92  
##  Mean   :111.70  
##  3rd Qu.:128.68  
##  Max.   :318.34  
## 
```

```r
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training<- Wage[inTrain,]
testing<- Wage[-inTrain,]
dim(training)
```

```
## [1] 2102   12
```

```r
dim(testing)
```

```
## [1] 898  12
```

```r
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot="pairs")
```

![](lecture_2_files/figure-html/unnamed-chunk-4-1.png) 

```r
qplot(age, wage, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-4-2.png) 

```r
qplot(age, wage, colour=jobclass, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-4-3.png) 

```r
qplot(age, wage, colour=education, data=training) + geom_smooth(method="lm", formula=y~x)
```

![](lecture_2_files/figure-html/unnamed-chunk-4-4.png) 

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(pander)
cutWage<- cut2(training$wage, g=3)
pander(table(cutWage))
```


-----------------------------------------------
 [ 20.1, 91.7)   [ 91.7,118.9)   [118.9,318.3] 
--------------- --------------- ---------------
      704             718             680      
-----------------------------------------------

```r
q1<- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot"))
q2<- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot", "jitter"))
library(grid)
library(gridExtra)
grid.arrange(q1, q2, ncol=2)
```

![](lecture_2_files/figure-html/unnamed-chunk-4-5.png) 

```r
t1<- table(cutWage, training$jobclass)
pander(t1)
```


----------------------------------------------------
      &nbsp;         1. Industrial   2. Information 
------------------- --------------- ----------------
 **[ 20.1, 91.7)**        447             257       

 **[ 91.7,118.9)**        363             355       

 **[118.9,318.3]**        277             403       
----------------------------------------------------

```r
pander(prop.table(t1,1))#proportion in each row, 2 would have been column
```


----------------------------------------------------
      &nbsp;         1. Industrial   2. Information 
------------------- --------------- ----------------
 **[ 20.1, 91.7)**      0.6349           0.3651     

 **[ 91.7,118.9)**      0.5056           0.4944     

 **[118.9,318.3]**      0.4074           0.5926     
----------------------------------------------------

```r
qplot(wage, colour=education, data=training, geom="density")
```

![](lecture_2_files/figure-html/unnamed-chunk-4-6.png) 

##Notes

* Make your plots only in trainig set
    * don't use the test set for exploration!
* Things you should be looking for 
    * imbalance in outcomes/predictors
    * outliers
    * groups of points not explained by a predictor
    * skewed variables

#Preprocessing


```r
library(caret)
library(kernlab)
data("spam")
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve, main = "", xlab = "ave. capital run length")
```

![](lecture_2_files/figure-html/unnamed-chunk-5-1.png) 

```r
mean(training$capitalAve)
```

```
## [1] 5.309768
```

```r
sd(training$capitalAve)
```

```
## [1] 31.43004
```

```r
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(trainCapAveS)
```

```
## [1] 7.473785e-18
```

```r
sd(trainCapAveS)
```

```
## [1] 1
```

```r
preObj<- preProcess(training[, -58], method = c("center", "scale"))
trainCapAveS<- predict(preObj, training[-58])$capitalAve
testCapAveS <- predict(preObj, testing[,-58])$capitalAve
mean(testCapAveS)
```

```
## [1] -0.01505296
```

```r
sd(testCapAveS)
```

```
## [1] 1.03795
```

```r
set.seed(32343)
modelFit<- train(type ~., data=training,
                 preProcess=c("center","scale"),
                 method="glm")
modelFit
```

```
## Generalized Linear Model 
## 
## 3451 samples
##   57 predictor
##    2 classes: 'nonspam', 'spam' 
## 
## Pre-processing: centered (57), scaled (57) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 3451, 3451, 3451, 3451, 3451, 3451, ... 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.9177634  0.8266828  0.006656576  0.01386059
## 
## 
```

```r
#removing suspiciously variable or very volatile predictors
preObj<- preProcess(training[,-58], method = c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2))
hist(trainCapAveS)
qqnorm(trainCapAveS)
```

![](lecture_2_files/figure-html/unnamed-chunk-5-2.png) 

```r
set.seed(13343)
#Make some values NA
training$capAve<- training$capitalAve
selectNA<- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA]<- NA

#Impute and standardize
preObj<- preProcess(training[,-58], method="knnImpute")
capAve<- predict(preObj, training[,-58])$capAve

#Standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)
```

* Traning and tests must be processed in the same way
* Test transformations will likely be imperfect
    * especially if the test/training sets collected at different times
* Careful when transforming factor varibles!

#Covariate creation

1. Level 1: From raw data to covariate
2. Level 2: Transforming tidy covariates



```r
library(kernlab)
data(spam)
spam$capitalAveSq<- spam$capitalAve^2
```

##Level 1, Raw data -> covariates

* Depends heavily on an application
* The balancing act is summarization vs. infromation loss
* Examples:
    * text files: frequency of words, frequency of phrases (Google ngrams), frequency of capital letters.
    * images: edges, corners, blobs, ridges (computer vision, feature detection)
    * webpages: number and type of images, position of elements, colors, videos (A/B Testing)
* The more knowledge of the system you have the better the job you will do.
* When in doubt, err on the side of more features
* Can be automated, but use caution!

##Level 2, Tidy covariates -> new covariates

* More necessary for some methods (regression, svms) that for others (classification trees)
* Should be done *only* on the training set
* The best approach is through exploratory analysis (plotting and tables)
* New covariates should be added to data frames


```r
library(ISLR)
library(caret)
data("Wage")
inTrain<- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training<- Wage[inTrain,]
testing<- Wage[-inTrain,]
```

Basic idea is to turn factor variables into indicator variables.


```r
table(training$jobclass)
```

```
## 
##  1. Industrial 2. Information 
##           1051           1051
```

```r
dummies<- dummyVars(wage ~ jobclass, data=training)
head(predict(dummies, newdata=training))
```

```
##        jobclass.1. Industrial jobclass.2. Information
## 86582                       0                       1
## 161300                      1                       0
## 155159                      0                       1
## 11443                       0                       1
## 376662                      0                       1
## 450601                      1                       0
```

```r
nsv<- nearZeroVar(training, saveMetrics = TRUE)
nsv
```

```
##            freqRatio percentUnique zeroVar   nzv
## year        1.037356    0.33301618   FALSE FALSE
## age         1.027027    2.85442436   FALSE FALSE
## sex         0.000000    0.04757374    TRUE  TRUE
## maritl      3.272931    0.23786870   FALSE FALSE
## race        8.938776    0.19029496   FALSE FALSE
## education   1.389002    0.23786870   FALSE FALSE
## region      0.000000    0.04757374    TRUE  TRUE
## jobclass    1.000000    0.09514748   FALSE FALSE
## health      2.468647    0.09514748   FALSE FALSE
## health_ins  2.352472    0.09514748   FALSE FALSE
## logwage     1.061728   19.17221694   FALSE FALSE
## wage        1.061728   19.17221694   FALSE FALSE
```

```r
library(splines)
bsBasis <- bs(training$age, df=3)
bsBasis
```

```
##                   1            2            3
##    [1,] 0.236850055 0.0253767916 9.063140e-04
##    [2,] 0.416337988 0.3211750193 8.258786e-02
##    [3,] 0.430813836 0.2910904300 6.556091e-02
##    [4,] 0.362525595 0.3866939680 1.374912e-01
##    [5,] 0.306334128 0.4241549461 1.957638e-01
##    [6,] 0.424154946 0.3063341278 7.374710e-02
##    [7,] 0.377630828 0.0906313987 7.250512e-03
##    [8,] 0.444358195 0.2275981001 3.885821e-02
##    [9,] 0.442218287 0.1953987782 2.877966e-02
##   [10,] 0.362525595 0.3866939680 1.374912e-01
##   [11,] 0.275519452 0.4362391326 2.302373e-01
##   [12,] 0.444093854 0.2114732637 3.356718e-02
##   [13,] 0.443086838 0.2436977611 4.467792e-02
##   [14,] 0.375000000 0.3750000000 1.250000e-01
##   [15,] 0.430813836 0.2910904300 6.556091e-02
##   [16,] 0.426168977 0.1482326877 1.718640e-02
##   [17,] 0.000000000 0.0000000000 0.000000e+00
##   [18,] 0.291090430 0.4308138364 2.125348e-01
##   [19,] 0.349346279 0.3975319727 1.507880e-01
##   [20,] 0.417093250 0.1331148669 1.416116e-02
##   [21,] 0.426168977 0.1482326877 1.718640e-02
##   [22,] 0.438655970 0.1794501695 2.447048e-02
##   [23,] 0.275519452 0.4362391326 2.302373e-01
##   [24,] 0.266544426 0.0339238361 1.439193e-03
##   [25,] 0.406028666 0.1184250277 1.151354e-02
##   [26,] 0.318229499 0.0540389715 3.058810e-03
##   [27,] 0.340371253 0.0654560102 4.195898e-03
##   [28,] 0.318229499 0.0540389715 3.058810e-03
##   [29,] 0.430813836 0.2910904300 6.556091e-02
##   [30,] 0.362525595 0.3866939680 1.374912e-01
##   [31,] 0.444358195 0.2275981001 3.885821e-02
##   [32,] 0.259696720 0.4403553087 2.488965e-01
##   [33,] 0.266544426 0.0339238361 1.439193e-03
##   [34,] 0.430813836 0.2910904300 6.556091e-02
##   [35,] 0.204487093 0.0179374643 5.244873e-04
##   [36,] 0.377630828 0.0906313987 7.250512e-03
##   [37,] 0.195398778 0.4422182874 3.336033e-01
##   [38,] 0.426168977 0.1482326877 1.718640e-02
##   [39,] 0.077678661 0.3601465208 5.565901e-01
##   [40,] 0.386693968 0.3625255950 1.132892e-01
##   [41,] 0.375000000 0.3750000000 1.250000e-01
##   [42,] 0.436239133 0.2755194522 5.800410e-02
##   [43,] 0.442218287 0.1953987782 2.877966e-02
##   [44,] 0.131453291 0.0066840657 1.132892e-04
##   [45,] 0.243697761 0.4430868383 2.685375e-01
##   [46,] 0.266544426 0.0339238361 1.439193e-03
##   [47,] 0.443086838 0.2436977611 4.467792e-02
##   [48,] 0.424154946 0.3063341278 7.374710e-02
##   [49,] 0.424154946 0.3063341278 7.374710e-02
##   [50,] 0.195398778 0.4422182874 3.336033e-01
##   [51,] 0.291090430 0.4308138364 2.125348e-01
##   [52,] 0.436239133 0.2755194522 5.800410e-02
##   [53,] 0.266544426 0.0339238361 1.439193e-03
##   [54,] 0.321175019 0.4163379880 1.798991e-01
##   [55,] 0.397531973 0.3493462791 1.023338e-01
##   [56,] 0.407438488 0.3355375785 9.210835e-02
##   [57,] 0.426168977 0.1482326877 1.718640e-02
##   [58,] 0.169380014 0.0116813803 2.685375e-04
##   [59,] 0.416337988 0.3211750193 8.258786e-02
##   [60,] 0.179450170 0.4386559699 3.574234e-01
##   [61,] 0.306334128 0.4241549461 1.957638e-01
##   [62,] 0.426168977 0.1482326877 1.718640e-02
##   [63,] 0.362525595 0.3866939680 1.374912e-01
##   [64,] 0.407438488 0.3355375785 9.210835e-02
##   [65,] 0.440355309 0.2596967205 5.105149e-02
##   [66,] 0.444093854 0.2114732637 3.356718e-02
##   [67,] 0.433331375 0.1637029640 2.061445e-02
##   [68,] 0.118425028 0.4060286664 4.640328e-01
##   [69,] 0.442218287 0.1953987782 2.877966e-02
##   [70,] 0.444358195 0.2275981001 3.885821e-02
##   [71,] 0.436239133 0.2755194522 5.800410e-02
##   [72,] 0.349346279 0.3975319727 1.507880e-01
##   [73,] 0.444093854 0.2114732637 3.356718e-02
##   [74,] 0.375000000 0.3750000000 1.250000e-01
##   [75,] 0.436239133 0.2755194522 5.800410e-02
##   [76,] 0.430813836 0.2910904300 6.556091e-02
##   [77,] 0.227598100 0.4443581954 2.891855e-01
##   [78,] 0.259696720 0.4403553087 2.488965e-01
##   [79,] 0.266544426 0.0339238361 1.439193e-03
##   [80,] 0.375000000 0.3750000000 1.250000e-01
##   [81,] 0.444093854 0.2114732637 3.356718e-02
##   [82,] 0.195398778 0.4422182874 3.336033e-01
##   [83,] 0.335537578 0.4074384881 1.649156e-01
##   [84,] 0.211473264 0.4440938538 3.108657e-01
##   [85,] 0.407438488 0.3355375785 9.210835e-02
##   [86,] 0.131453291 0.0066840657 1.132892e-04
##   [87,] 0.195398778 0.4422182874 3.336033e-01
##   [88,] 0.406028666 0.1184250277 1.151354e-02
##   [89,] 0.243697761 0.4430868383 2.685375e-01
##   [90,] 0.406028666 0.1184250277 1.151354e-02
##   [91,] 0.169380014 0.0116813803 2.685375e-04
##   [92,] 0.349346279 0.3975319727 1.507880e-01
##   [93,] 0.424154946 0.3063341278 7.374710e-02
##   [94,] 0.211473264 0.4440938538 3.108657e-01
##   [95,] 0.443086838 0.2436977611 4.467792e-02
##   [96,] 0.433331375 0.1637029640 2.061445e-02
##   [97,] 0.433331375 0.1637029640 2.061445e-02
##   [98,] 0.211473264 0.4440938538 3.108657e-01
##   [99,] 0.444093854 0.2114732637 3.356718e-02
##  [100,] 0.321175019 0.4163379880 1.798991e-01
##  [101,] 0.259696720 0.4403553087 2.488965e-01
##  [102,] 0.148232688 0.4261689772 4.084119e-01
##  [103,] 0.433331375 0.1637029640 2.061445e-02
##  [104,] 0.306334128 0.4241549461 1.957638e-01
##  [105,] 0.416337988 0.3211750193 8.258786e-02
##  [106,] 0.243697761 0.4430868383 2.685375e-01
##  [107,] 0.386693968 0.3625255950 1.132892e-01
##  [108,] 0.407438488 0.3355375785 9.210835e-02
##  [109,] 0.407438488 0.3355375785 9.210835e-02
##  [110,] 0.291090430 0.4308138364 2.125348e-01
##  [111,] 0.349346279 0.3975319727 1.507880e-01
##  [112,] 0.375000000 0.3750000000 1.250000e-01
##  [113,] 0.426168977 0.1482326877 1.718640e-02
##  [114,] 0.321175019 0.4163379880 1.798991e-01
##  [115,] 0.443086838 0.2436977611 4.467792e-02
##  [116,] 0.362525595 0.3866939680 1.374912e-01
##  [117,] 0.444358195 0.2275981001 3.885821e-02
##  [118,] 0.335537578 0.4074384881 1.649156e-01
##  [119,] 0.362525595 0.3866939680 1.374912e-01
##  [120,] 0.386693968 0.3625255950 1.132892e-01
##  [121,] 0.397531973 0.3493462791 1.023338e-01
##  [122,] 0.444358195 0.2275981001 3.885821e-02
##  [123,] 0.424154946 0.3063341278 7.374710e-02
##  [124,] 0.442218287 0.1953987782 2.877966e-02
##  [125,] 0.335537578 0.4074384881 1.649156e-01
##  [126,] 0.293645732 0.0435030714 2.148300e-03
##  [127,] 0.392899701 0.1042386963 9.218388e-03
##  [128,] 0.243697761 0.4430868383 2.685375e-01
##  [129,] 0.377630828 0.0906313987 7.250512e-03
##  [130,] 0.318229499 0.0540389715 3.058810e-03
##  [131,] 0.443086838 0.2436977611 4.467792e-02
##  [132,] 0.291090430 0.4308138364 2.125348e-01
##  [133,] 0.433331375 0.1637029640 2.061445e-02
##  [134,] 0.360146521 0.0776786613 5.584740e-03
##  [135,] 0.266544426 0.0339238361 1.439193e-03
##  [136,] 0.443086838 0.2436977611 4.467792e-02
##  [137,] 0.318229499 0.0540389715 3.058810e-03
##  [138,] 0.375000000 0.3750000000 1.250000e-01
##  [139,] 0.169380014 0.0116813803 2.685375e-04
##  [140,] 0.375000000 0.3750000000 1.250000e-01
##  [141,] 0.266544426 0.0339238361 1.439193e-03
##  [142,] 0.360146521 0.0776786613 5.584740e-03
##  [143,] 0.442218287 0.1953987782 2.877966e-02
##  [144,] 0.433331375 0.1637029640 2.061445e-02
##  [145,] 0.243697761 0.4430868383 2.685375e-01
##  [146,] 0.444358195 0.2275981001 3.885821e-02
##  [147,] 0.440355309 0.2596967205 5.105149e-02
##  [148,] 0.442218287 0.1953987782 2.877966e-02
##  [149,] 0.179450170 0.4386559699 3.574234e-01
##  [150,] 0.318229499 0.0540389715 3.058810e-03
##  [151,] 0.442218287 0.1953987782 2.877966e-02
##  [152,] 0.275519452 0.4362391326 2.302373e-01
##  [153,] 0.438655970 0.1794501695 2.447048e-02
##  [154,] 0.204487093 0.0179374643 5.244873e-04
##  [155,] 0.407438488 0.3355375785 9.210835e-02
##  [156,] 0.293645732 0.0435030714 2.148300e-03
##  [157,] 0.430813836 0.2910904300 6.556091e-02
##  [158,] 0.438655970 0.1794501695 2.447048e-02
##  [159,] 0.306334128 0.4241549461 1.957638e-01
##  [160,] 0.443086838 0.2436977611 4.467792e-02
##  [161,] 0.426168977 0.1482326877 1.718640e-02
##  [162,] 0.430813836 0.2910904300 6.556091e-02
##  [163,] 0.227598100 0.4443581954 2.891855e-01
##  [164,] 0.211473264 0.4440938538 3.108657e-01
##  [165,] 0.375000000 0.3750000000 1.250000e-01
##  [166,] 0.416337988 0.3211750193 8.258786e-02
##  [167,] 0.426168977 0.1482326877 1.718640e-02
##  [168,] 0.169380014 0.0116813803 2.685375e-04
##  [169,] 0.443086838 0.2436977611 4.467792e-02
##  [170,] 0.440355309 0.2596967205 5.105149e-02
##  [171,] 0.438655970 0.1794501695 2.447048e-02
##  [172,] 0.397531973 0.3493462791 1.023338e-01
##  [173,] 0.433331375 0.1637029640 2.061445e-02
##  [174,] 0.443086838 0.2436977611 4.467792e-02
##  [175,] 0.259696720 0.4403553087 2.488965e-01
##  [176,] 0.033923836 0.2665444262 6.980925e-01
##  [177,] 0.360146521 0.0776786613 5.584740e-03
##  [178,] 0.377630828 0.0906313987 7.250512e-03
##  [179,] 0.360146521 0.0776786613 5.584740e-03
##  [180,] 0.438655970 0.1794501695 2.447048e-02
##  [181,] 0.444358195 0.2275981001 3.885821e-02
##  [182,] 0.386693968 0.3625255950 1.132892e-01
##  [183,] 0.416337988 0.3211750193 8.258786e-02
##  [184,] 0.362525595 0.3866939680 1.374912e-01
##  [185,] 0.243697761 0.4430868383 2.685375e-01
##  [186,] 0.386693968 0.3625255950 1.132892e-01
##  [187,] 0.440355309 0.2596967205 5.105149e-02
##  [188,] 0.318229499 0.0540389715 3.058810e-03
##  [189,] 0.424154946 0.3063341278 7.374710e-02
##  [190,] 0.406028666 0.1184250277 1.151354e-02
##  [191,] 0.407438488 0.3355375785 9.210835e-02
##  [192,] 0.169380014 0.0116813803 2.685375e-04
##  [193,] 0.321175019 0.4163379880 1.798991e-01
##  [194,] 0.426168977 0.1482326877 1.718640e-02
##  [195,] 0.444093854 0.2114732637 3.356718e-02
##  [196,] 0.266544426 0.0339238361 1.439193e-03
##  [197,] 0.360146521 0.0776786613 5.584740e-03
##  [198,] 0.340371253 0.0654560102 4.195898e-03
##  [199,] 0.291090430 0.4308138364 2.125348e-01
##  [200,] 0.275519452 0.4362391326 2.302373e-01
##  [201,] 0.195398778 0.4422182874 3.336033e-01
##  [202,] 0.397531973 0.3493462791 1.023338e-01
##  [203,] 0.335537578 0.4074384881 1.649156e-01
##  [204,] 0.417093250 0.1331148669 1.416116e-02
##  [205,] 0.243697761 0.4430868383 2.685375e-01
##  [206,] 0.318229499 0.0540389715 3.058810e-03
##  [207,] 0.335537578 0.4074384881 1.649156e-01
##  [208,] 0.416337988 0.3211750193 8.258786e-02
##  [209,] 0.169380014 0.0116813803 2.685375e-04
##  [210,] 0.266544426 0.0339238361 1.439193e-03
##  [211,] 0.438655970 0.1794501695 2.447048e-02
##  [212,] 0.392899701 0.1042386963 9.218388e-03
##  [213,] 0.335537578 0.4074384881 1.649156e-01
##  [214,] 0.407438488 0.3355375785 9.210835e-02
##  [215,] 0.416337988 0.3211750193 8.258786e-02
##  [216,] 0.443086838 0.2436977611 4.467792e-02
##  [217,] 0.436239133 0.2755194522 5.800410e-02
##  [218,] 0.440355309 0.2596967205 5.105149e-02
##  [219,] 0.266544426 0.0339238361 1.439193e-03
##  [220,] 0.236850055 0.0253767916 9.063140e-04
##  [221,] 0.349346279 0.3975319727 1.507880e-01
##  [222,] 0.440355309 0.2596967205 5.105149e-02
##  [223,] 0.377630828 0.0906313987 7.250512e-03
##  [224,] 0.291090430 0.4308138364 2.125348e-01
##  [225,] 0.204487093 0.0179374643 5.244873e-04
##  [226,] 0.211473264 0.4440938538 3.108657e-01
##  [227,] 0.443086838 0.2436977611 4.467792e-02
##  [228,] 0.000000000 0.0000000000 1.000000e+00
##  [229,] 0.443086838 0.2436977611 4.467792e-02
##  [230,] 0.433331375 0.1637029640 2.061445e-02
##  [231,] 0.291090430 0.4308138364 2.125348e-01
##  [232,] 0.236850055 0.0253767916 9.063140e-04
##  [233,] 0.444358195 0.2275981001 3.885821e-02
##  [234,] 0.377630828 0.0906313987 7.250512e-03
##  [235,] 0.090631399 0.3776308281 5.244873e-01
##  [236,] 0.306334128 0.4241549461 1.957638e-01
##  [237,] 0.318229499 0.0540389715 3.058810e-03
##  [238,] 0.426168977 0.1482326877 1.718640e-02
##  [239,] 0.321175019 0.4163379880 1.798991e-01
##  [240,] 0.227598100 0.4443581954 2.891855e-01
##  [241,] 0.416337988 0.3211750193 8.258786e-02
##  [242,] 0.430813836 0.2910904300 6.556091e-02
##  [243,] 0.377630828 0.0906313987 7.250512e-03
##  [244,] 0.436239133 0.2755194522 5.800410e-02
##  [245,] 0.204487093 0.0179374643 5.244873e-04
##  [246,] 0.243697761 0.4430868383 2.685375e-01
##  [247,] 0.417093250 0.1331148669 1.416116e-02
##  [248,] 0.275519452 0.4362391326 2.302373e-01
##  [249,] 0.442218287 0.1953987782 2.877966e-02
##  [250,] 0.417093250 0.1331148669 1.416116e-02
##  [251,] 0.362525595 0.3866939680 1.374912e-01
##  [252,] 0.430813836 0.2910904300 6.556091e-02
##  [253,] 0.321175019 0.4163379880 1.798991e-01
##  [254,] 0.442218287 0.1953987782 2.877966e-02
##  [255,] 0.090631399 0.0030210466 3.356718e-05
##  [256,] 0.293645732 0.0435030714 2.148300e-03
##  [257,] 0.360146521 0.0776786613 5.584740e-03
##  [258,] 0.259696720 0.4403553087 2.488965e-01
##  [259,] 0.397531973 0.3493462791 1.023338e-01
##  [260,] 0.444093854 0.2114732637 3.356718e-02
##  [261,] 0.204487093 0.0179374643 5.244873e-04
##  [262,] 0.392899701 0.1042386963 9.218388e-03
##  [263,] 0.430813836 0.2910904300 6.556091e-02
##  [264,] 0.417093250 0.1331148669 1.416116e-02
##  [265,] 0.386693968 0.3625255950 1.132892e-01
##  [266,] 0.377630828 0.0906313987 7.250512e-03
##  [267,] 0.424154946 0.3063341278 7.374710e-02
##  [268,] 0.444093854 0.2114732637 3.356718e-02
##  [269,] 0.397531973 0.3493462791 1.023338e-01
##  [270,] 0.340371253 0.0654560102 4.195898e-03
##  [271,] 0.204487093 0.0179374643 5.244873e-04
##  [272,] 0.318229499 0.0540389715 3.058810e-03
##  [273,] 0.417093250 0.1331148669 1.416116e-02
##  [274,] 0.375000000 0.3750000000 1.250000e-01
##  [275,] 0.318229499 0.0540389715 3.058810e-03
##  [276,] 0.386693968 0.3625255950 1.132892e-01
##  [277,] 0.444093854 0.2114732637 3.356718e-02
##  [278,] 0.243697761 0.4430868383 2.685375e-01
##  [279,] 0.407438488 0.3355375785 9.210835e-02
##  [280,] 0.321175019 0.4163379880 1.798991e-01
##  [281,] 0.436239133 0.2755194522 5.800410e-02
##  [282,] 0.443086838 0.2436977611 4.467792e-02
##  [283,] 0.433331375 0.1637029640 2.061445e-02
##  [284,] 0.362525595 0.3866939680 1.374912e-01
##  [285,] 0.426168977 0.1482326877 1.718640e-02
##  [286,] 0.386693968 0.3625255950 1.132892e-01
##  [287,] 0.375000000 0.3750000000 1.250000e-01
##  [288,] 0.440355309 0.2596967205 5.105149e-02
##  [289,] 0.243697761 0.4430868383 2.685375e-01
##  [290,] 0.362525595 0.3866939680 1.374912e-01
##  [291,] 0.444093854 0.2114732637 3.356718e-02
##  [292,] 0.377630828 0.0906313987 7.250512e-03
##  [293,] 0.424154946 0.3063341278 7.374710e-02
##  [294,] 0.243697761 0.4430868383 2.685375e-01
##  [295,] 0.416337988 0.3211750193 8.258786e-02
##  [296,] 0.424154946 0.3063341278 7.374710e-02
##  [297,] 0.416337988 0.3211750193 8.258786e-02
##  [298,] 0.349346279 0.3975319727 1.507880e-01
##  [299,] 0.195398778 0.4422182874 3.336033e-01
##  [300,] 0.416337988 0.3211750193 8.258786e-02
##  [301,] 0.426168977 0.1482326877 1.718640e-02
##  [302,] 0.417093250 0.1331148669 1.416116e-02
##  [303,] 0.362525595 0.3866939680 1.374912e-01
##  [304,] 0.362525595 0.3866939680 1.374912e-01
##  [305,] 0.444358195 0.2275981001 3.885821e-02
##  [306,] 0.293645732 0.0435030714 2.148300e-03
##  [307,] 0.375000000 0.3750000000 1.250000e-01
##  [308,] 0.392899701 0.1042386963 9.218388e-03
##  [309,] 0.275519452 0.4362391326 2.302373e-01
##  [310,] 0.195398778 0.4422182874 3.336033e-01
##  [311,] 0.275519452 0.4362391326 2.302373e-01
##  [312,] 0.349346279 0.3975319727 1.507880e-01
##  [313,] 0.436239133 0.2755194522 5.800410e-02
##  [314,] 0.416337988 0.3211750193 8.258786e-02
##  [315,] 0.386693968 0.3625255950 1.132892e-01
##  [316,] 0.417093250 0.1331148669 1.416116e-02
##  [317,] 0.392899701 0.1042386963 9.218388e-03
##  [318,] 0.386693968 0.3625255950 1.132892e-01
##  [319,] 0.433331375 0.1637029640 2.061445e-02
##  [320,] 0.397531973 0.3493462791 1.023338e-01
##  [321,] 0.416337988 0.3211750193 8.258786e-02
##  [322,] 0.407438488 0.3355375785 9.210835e-02
##  [323,] 0.397531973 0.3493462791 1.023338e-01
##  [324,] 0.375000000 0.3750000000 1.250000e-01
##  [325,] 0.438655970 0.1794501695 2.447048e-02
##  [326,] 0.349346279 0.3975319727 1.507880e-01
##  [327,] 0.407438488 0.3355375785 9.210835e-02
##  [328,] 0.430813836 0.2910904300 6.556091e-02
##  [329,] 0.424154946 0.3063341278 7.374710e-02
##  [330,] 0.195398778 0.4422182874 3.336033e-01
##  [331,] 0.442218287 0.1953987782 2.877966e-02
##  [332,] 0.444093854 0.2114732637 3.356718e-02
##  [333,] 0.440355309 0.2596967205 5.105149e-02
##  [334,] 0.377630828 0.0906313987 7.250512e-03
##  [335,] 0.349346279 0.3975319727 1.507880e-01
##  [336,] 0.433331375 0.1637029640 2.061445e-02
##  [337,] 0.318229499 0.0540389715 3.058810e-03
##  [338,] 0.349346279 0.3975319727 1.507880e-01
##  [339,] 0.440355309 0.2596967205 5.105149e-02
##  [340,] 0.163702964 0.4333313752 3.823512e-01
##  [341,] 0.340371253 0.0654560102 4.195898e-03
##  [342,] 0.362525595 0.3866939680 1.374912e-01
##  [343,] 0.440355309 0.2596967205 5.105149e-02
##  [344,] 0.204487093 0.0179374643 5.244873e-04
##  [345,] 0.416337988 0.3211750193 8.258786e-02
##  [346,] 0.163702964 0.4333313752 3.823512e-01
##  [347,] 0.227598100 0.4443581954 2.891855e-01
##  [348,] 0.377630828 0.0906313987 7.250512e-03
##  [349,] 0.416337988 0.3211750193 8.258786e-02
##  [350,] 0.335537578 0.4074384881 1.649156e-01
##  [351,] 0.306334128 0.4241549461 1.957638e-01
##  [352,] 0.377630828 0.0906313987 7.250512e-03
##  [353,] 0.397531973 0.3493462791 1.023338e-01
##  [354,] 0.397531973 0.3493462791 1.023338e-01
##  [355,] 0.444358195 0.2275981001 3.885821e-02
##  [356,] 0.362525595 0.3866939680 1.374912e-01
##  [357,] 0.397531973 0.3493462791 1.023338e-01
##  [358,] 0.416337988 0.3211750193 8.258786e-02
##  [359,] 0.424154946 0.3063341278 7.374710e-02
##  [360,] 0.436239133 0.2755194522 5.800410e-02
##  [361,] 0.275519452 0.4362391326 2.302373e-01
##  [362,] 0.362525595 0.3866939680 1.374912e-01
##  [363,] 0.321175019 0.4163379880 1.798991e-01
##  [364,] 0.444093854 0.2114732637 3.356718e-02
##  [365,] 0.275519452 0.4362391326 2.302373e-01
##  [366,] 0.362525595 0.3866939680 1.374912e-01
##  [367,] 0.375000000 0.3750000000 1.250000e-01
##  [368,] 0.436239133 0.2755194522 5.800410e-02
##  [369,] 0.362525595 0.3866939680 1.374912e-01
##  [370,] 0.321175019 0.4163379880 1.798991e-01
##  [371,] 0.340371253 0.0654560102 4.195898e-03
##  [372,] 0.416337988 0.3211750193 8.258786e-02
##  [373,] 0.236850055 0.0253767916 9.063140e-04
##  [374,] 0.266544426 0.0339238361 1.439193e-03
##  [375,] 0.397531973 0.3493462791 1.023338e-01
##  [376,] 0.444093854 0.2114732637 3.356718e-02
##  [377,] 0.417093250 0.1331148669 1.416116e-02
##  [378,] 0.444358195 0.2275981001 3.885821e-02
##  [379,] 0.407438488 0.3355375785 9.210835e-02
##  [380,] 0.195398778 0.4422182874 3.336033e-01
##  [381,] 0.406028666 0.1184250277 1.151354e-02
##  [382,] 0.195398778 0.4422182874 3.336033e-01
##  [383,] 0.416337988 0.3211750193 8.258786e-02
##  [384,] 0.243697761 0.4430868383 2.685375e-01
##  [385,] 0.266544426 0.0339238361 1.439193e-03
##  [386,] 0.426168977 0.1482326877 1.718640e-02
##  [387,] 0.424154946 0.3063341278 7.374710e-02
##  [388,] 0.148232688 0.4261689772 4.084119e-01
##  [389,] 0.306334128 0.4241549461 1.957638e-01
##  [390,] 0.436239133 0.2755194522 5.800410e-02
##  [391,] 0.392899701 0.1042386963 9.218388e-03
##  [392,] 0.266544426 0.0339238361 1.439193e-03
##  [393,] 0.349346279 0.3975319727 1.507880e-01
##  [394,] 0.340371253 0.0654560102 4.195898e-03
##  [395,] 0.321175019 0.4163379880 1.798991e-01
##  [396,] 0.407438488 0.3355375785 9.210835e-02
##  [397,] 0.444093854 0.2114732637 3.356718e-02
##  [398,] 0.444358195 0.2275981001 3.885821e-02
##  [399,] 0.442218287 0.1953987782 2.877966e-02
##  [400,] 0.227598100 0.4443581954 2.891855e-01
##  [401,] 0.417093250 0.1331148669 1.416116e-02
##  [402,] 0.204487093 0.0179374643 5.244873e-04
##  [403,] 0.442218287 0.1953987782 2.877966e-02
##  [404,] 0.318229499 0.0540389715 3.058810e-03
##  [405,] 0.397531973 0.3493462791 1.023338e-01
##  [406,] 0.335537578 0.4074384881 1.649156e-01
##  [407,] 0.442218287 0.1953987782 2.877966e-02
##  [408,] 0.426168977 0.1482326877 1.718640e-02
##  [409,] 0.349346279 0.3975319727 1.507880e-01
##  [410,] 0.362525595 0.3866939680 1.374912e-01
##  [411,] 0.306334128 0.4241549461 1.957638e-01
##  [412,] 0.362525595 0.3866939680 1.374912e-01
##  [413,] 0.406028666 0.1184250277 1.151354e-02
##  [414,] 0.442218287 0.1953987782 2.877966e-02
##  [415,] 0.046838810 0.0007678494 4.195898e-06
##  [416,] 0.406028666 0.1184250277 1.151354e-02
##  [417,] 0.436239133 0.2755194522 5.800410e-02
##  [418,] 0.430813836 0.2910904300 6.556091e-02
##  [419,] 0.424154946 0.3063341278 7.374710e-02
##  [420,] 0.443086838 0.2436977611 4.467792e-02
##  [421,] 0.430813836 0.2910904300 6.556091e-02
##  [422,] 0.406028666 0.1184250277 1.151354e-02
##  [423,] 0.195398778 0.4422182874 3.336033e-01
##  [424,] 0.397531973 0.3493462791 1.023338e-01
##  [425,] 0.291090430 0.4308138364 2.125348e-01
##  [426,] 0.335537578 0.4074384881 1.649156e-01
##  [427,] 0.318229499 0.0540389715 3.058810e-03
##  [428,] 0.169380014 0.0116813803 2.685375e-04
##  [429,] 0.436239133 0.2755194522 5.800410e-02
##  [430,] 0.392899701 0.1042386963 9.218388e-03
##  [431,] 0.227598100 0.4443581954 2.891855e-01
##  [432,] 0.438655970 0.1794501695 2.447048e-02
##  [433,] 0.406028666 0.1184250277 1.151354e-02
##  [434,] 0.406028666 0.1184250277 1.151354e-02
##  [435,] 0.266544426 0.0339238361 1.439193e-03
##  [436,] 0.430813836 0.2910904300 6.556091e-02
##  [437,] 0.424154946 0.3063341278 7.374710e-02
##  [438,] 0.259696720 0.4403553087 2.488965e-01
##  [439,] 0.440355309 0.2596967205 5.105149e-02
##  [440,] 0.444093854 0.2114732637 3.356718e-02
##  [441,] 0.243697761 0.4430868383 2.685375e-01
##  [442,] 0.227598100 0.4443581954 2.891855e-01
##  [443,] 0.444358195 0.2275981001 3.885821e-02
##  [444,] 0.424154946 0.3063341278 7.374710e-02
##  [445,] 0.065456010 0.3403712531 5.899768e-01
##  [446,] 0.318229499 0.0540389715 3.058810e-03
##  [447,] 0.397531973 0.3493462791 1.023338e-01
##  [448,] 0.360146521 0.0776786613 5.584740e-03
##  [449,] 0.436239133 0.2755194522 5.800410e-02
##  [450,] 0.349346279 0.3975319727 1.507880e-01
##  [451,] 0.444358195 0.2275981001 3.885821e-02
##  [452,] 0.204487093 0.0179374643 5.244873e-04
##  [453,] 0.392899701 0.1042386963 9.218388e-03
##  [454,] 0.227598100 0.4443581954 2.891855e-01
##  [455,] 0.436239133 0.2755194522 5.800410e-02
##  [456,] 0.433331375 0.1637029640 2.061445e-02
##  [457,] 0.444093854 0.2114732637 3.356718e-02
##  [458,] 0.416337988 0.3211750193 8.258786e-02
##  [459,] 0.243697761 0.4430868383 2.685375e-01
##  [460,] 0.293645732 0.0435030714 2.148300e-03
##  [461,] 0.377630828 0.0906313987 7.250512e-03
##  [462,] 0.306334128 0.4241549461 1.957638e-01
##  [463,] 0.335537578 0.4074384881 1.649156e-01
##  [464,] 0.033923836 0.2665444262 6.980925e-01
##  [465,] 0.133114867 0.4170932496 4.356307e-01
##  [466,] 0.321175019 0.4163379880 1.798991e-01
##  [467,] 0.335537578 0.4074384881 1.649156e-01
##  [468,] 0.259696720 0.4403553087 2.488965e-01
##  [469,] 0.406028666 0.1184250277 1.151354e-02
##  [470,] 0.349346279 0.3975319727 1.507880e-01
##  [471,] 0.430813836 0.2910904300 6.556091e-02
##  [472,] 0.362525595 0.3866939680 1.374912e-01
##  [473,] 0.321175019 0.4163379880 1.798991e-01
##  [474,] 0.306334128 0.4241549461 1.957638e-01
##  [475,] 0.443086838 0.2436977611 4.467792e-02
##  [476,] 0.377630828 0.0906313987 7.250512e-03
##  [477,] 0.416337988 0.3211750193 8.258786e-02
##  [478,] 0.291090430 0.4308138364 2.125348e-01
##  [479,] 0.416337988 0.3211750193 8.258786e-02
##  [480,] 0.424154946 0.3063341278 7.374710e-02
##  [481,] 0.442218287 0.1953987782 2.877966e-02
##  [482,] 0.440355309 0.2596967205 5.105149e-02
##  [483,] 0.335537578 0.4074384881 1.649156e-01
##  [484,] 0.291090430 0.4308138364 2.125348e-01
##  [485,] 0.430813836 0.2910904300 6.556091e-02
##  [486,] 0.318229499 0.0540389715 3.058810e-03
##  [487,] 0.430813836 0.2910904300 6.556091e-02
##  [488,] 0.407438488 0.3355375785 9.210835e-02
##  [489,] 0.386693968 0.3625255950 1.132892e-01
##  [490,] 0.360146521 0.0776786613 5.584740e-03
##  [491,] 0.236850055 0.0253767916 9.063140e-04
##  [492,] 0.362525595 0.3866939680 1.374912e-01
##  [493,] 0.236850055 0.0253767916 9.063140e-04
##  [494,] 0.436239133 0.2755194522 5.800410e-02
##  [495,] 0.375000000 0.3750000000 1.250000e-01
##  [496,] 0.443086838 0.2436977611 4.467792e-02
##  [497,] 0.440355309 0.2596967205 5.105149e-02
##  [498,] 0.426168977 0.1482326877 1.718640e-02
##  [499,] 0.236850055 0.0253767916 9.063140e-04
##  [500,] 0.424154946 0.3063341278 7.374710e-02
##  [501,] 0.266544426 0.0339238361 1.439193e-03
##  [502,] 0.443086838 0.2436977611 4.467792e-02
##  [503,] 0.266544426 0.0339238361 1.439193e-03
##  [504,] 0.424154946 0.3063341278 7.374710e-02
##  [505,] 0.243697761 0.4430868383 2.685375e-01
##  [506,] 0.335537578 0.4074384881 1.649156e-01
##  [507,] 0.211473264 0.4440938538 3.108657e-01
##  [508,] 0.349346279 0.3975319727 1.507880e-01
##  [509,] 0.416337988 0.3211750193 8.258786e-02
##  [510,] 0.430813836 0.2910904300 6.556091e-02
##  [511,] 0.416337988 0.3211750193 8.258786e-02
##  [512,] 0.443086838 0.2436977611 4.467792e-02
##  [513,] 0.349346279 0.3975319727 1.507880e-01
##  [514,] 0.335537578 0.4074384881 1.649156e-01
##  [515,] 0.392899701 0.1042386963 9.218388e-03
##  [516,] 0.443086838 0.2436977611 4.467792e-02
##  [517,] 0.293645732 0.0435030714 2.148300e-03
##  [518,] 0.375000000 0.3750000000 1.250000e-01
##  [519,] 0.444093854 0.2114732637 3.356718e-02
##  [520,] 0.362525595 0.3866939680 1.374912e-01
##  [521,] 0.360146521 0.0776786613 5.584740e-03
##  [522,] 0.417093250 0.1331148669 1.416116e-02
##  [523,] 0.179450170 0.4386559699 3.574234e-01
##  [524,] 0.416337988 0.3211750193 8.258786e-02
##  [525,] 0.275519452 0.4362391326 2.302373e-01
##  [526,] 0.243697761 0.4430868383 2.685375e-01
##  [527,] 0.444358195 0.2275981001 3.885821e-02
##  [528,] 0.375000000 0.3750000000 1.250000e-01
##  [529,] 0.236850055 0.0253767916 9.063140e-04
##  [530,] 0.243697761 0.4430868383 2.685375e-01
##  [531,] 0.397531973 0.3493462791 1.023338e-01
##  [532,] 0.440355309 0.2596967205 5.105149e-02
##  [533,] 0.054038972 0.3182294988 6.246727e-01
##  [534,] 0.397531973 0.3493462791 1.023338e-01
##  [535,] 0.444093854 0.2114732637 3.356718e-02
##  [536,] 0.392899701 0.1042386963 9.218388e-03
##  [537,] 0.275519452 0.4362391326 2.302373e-01
##  [538,] 0.424154946 0.3063341278 7.374710e-02
##  [539,] 0.417093250 0.1331148669 1.416116e-02
##  [540,] 0.392899701 0.1042386963 9.218388e-03
##  [541,] 0.291090430 0.4308138364 2.125348e-01
##  [542,] 0.386693968 0.3625255950 1.132892e-01
##  [543,] 0.291090430 0.4308138364 2.125348e-01
##  [544,] 0.407438488 0.3355375785 9.210835e-02
##  [545,] 0.386693968 0.3625255950 1.132892e-01
##  [546,] 0.204487093 0.0179374643 5.244873e-04
##  [547,] 0.211473264 0.4440938538 3.108657e-01
##  [548,] 0.426168977 0.1482326877 1.718640e-02
##  [549,] 0.416337988 0.3211750193 8.258786e-02
##  [550,] 0.340371253 0.0654560102 4.195898e-03
##  [551,] 0.417093250 0.1331148669 1.416116e-02
##  [552,] 0.243697761 0.4430868383 2.685375e-01
##  [553,] 0.397531973 0.3493462791 1.023338e-01
##  [554,] 0.236850055 0.0253767916 9.063140e-04
##  [555,] 0.275519452 0.4362391326 2.302373e-01
##  [556,] 0.275519452 0.4362391326 2.302373e-01
##  [557,] 0.204487093 0.0179374643 5.244873e-04
##  [558,] 0.416337988 0.3211750193 8.258786e-02
##  [559,] 0.243697761 0.4430868383 2.685375e-01
##  [560,] 0.377630828 0.0906313987 7.250512e-03
##  [561,] 0.386693968 0.3625255950 1.132892e-01
##  [562,] 0.442218287 0.1953987782 2.877966e-02
##  [563,] 0.375000000 0.3750000000 1.250000e-01
##  [564,] 0.392899701 0.1042386963 9.218388e-03
##  [565,] 0.335537578 0.4074384881 1.649156e-01
##  [566,] 0.065456010 0.3403712531 5.899768e-01
##  [567,] 0.426168977 0.1482326877 1.718640e-02
##  [568,] 0.444093854 0.2114732637 3.356718e-02
##  [569,] 0.340371253 0.0654560102 4.195898e-03
##  [570,] 0.444093854 0.2114732637 3.356718e-02
##  [571,] 0.444358195 0.2275981001 3.885821e-02
##  [572,] 0.335537578 0.4074384881 1.649156e-01
##  [573,] 0.426168977 0.1482326877 1.718640e-02
##  [574,] 0.417093250 0.1331148669 1.416116e-02
##  [575,] 0.243697761 0.4430868383 2.685375e-01
##  [576,] 0.444093854 0.2114732637 3.356718e-02
##  [577,] 0.444093854 0.2114732637 3.356718e-02
##  [578,] 0.392899701 0.1042386963 9.218388e-03
##  [579,] 0.321175019 0.4163379880 1.798991e-01
##  [580,] 0.131453291 0.0066840657 1.132892e-04
##  [581,] 0.444093854 0.2114732637 3.356718e-02
##  [582,] 0.340371253 0.0654560102 4.195898e-03
##  [583,] 0.406028666 0.1184250277 1.151354e-02
##  [584,] 0.340371253 0.0654560102 4.195898e-03
##  [585,] 0.436239133 0.2755194522 5.800410e-02
##  [586,] 0.340371253 0.0654560102 4.195898e-03
##  [587,] 0.386693968 0.3625255950 1.132892e-01
##  [588,] 0.291090430 0.4308138364 2.125348e-01
##  [589,] 0.442218287 0.1953987782 2.877966e-02
##  [590,] 0.090631399 0.3776308281 5.244873e-01
##  [591,] 0.133114867 0.4170932496 4.356307e-01
##  [592,] 0.442218287 0.1953987782 2.877966e-02
##  [593,] 0.417093250 0.1331148669 1.416116e-02
##  [594,] 0.046838810 0.0007678494 4.195898e-06
##  [595,] 0.362525595 0.3866939680 1.374912e-01
##  [596,] 0.443086838 0.2436977611 4.467792e-02
##  [597,] 0.118425028 0.4060286664 4.640328e-01
##  [598,] 0.433331375 0.1637029640 2.061445e-02
##  [599,] 0.417093250 0.1331148669 1.416116e-02
##  [600,] 0.424154946 0.3063341278 7.374710e-02
##  [601,] 0.397531973 0.3493462791 1.023338e-01
##  [602,] 0.291090430 0.4308138364 2.125348e-01
##  [603,] 0.417093250 0.1331148669 1.416116e-02
##  [604,] 0.275519452 0.4362391326 2.302373e-01
##  [605,] 0.397531973 0.3493462791 1.023338e-01
##  [606,] 0.416337988 0.3211750193 8.258786e-02
##  [607,] 0.424154946 0.3063341278 7.374710e-02
##  [608,] 0.266544426 0.0339238361 1.439193e-03
##  [609,] 0.416337988 0.3211750193 8.258786e-02
##  [610,] 0.275519452 0.4362391326 2.302373e-01
##  [611,] 0.397531973 0.3493462791 1.023338e-01
##  [612,] 0.444358195 0.2275981001 3.885821e-02
##  [613,] 0.386693968 0.3625255950 1.132892e-01
##  [614,] 0.436239133 0.2755194522 5.800410e-02
##  [615,] 0.291090430 0.4308138364 2.125348e-01
##  [616,] 0.195398778 0.4422182874 3.336033e-01
##  [617,] 0.444358195 0.2275981001 3.885821e-02
##  [618,] 0.377630828 0.0906313987 7.250512e-03
##  [619,] 0.375000000 0.3750000000 1.250000e-01
##  [620,] 0.417093250 0.1331148669 1.416116e-02
##  [621,] 0.392899701 0.1042386963 9.218388e-03
##  [622,] 0.291090430 0.4308138364 2.125348e-01
##  [623,] 0.438655970 0.1794501695 2.447048e-02
##  [624,] 0.417093250 0.1331148669 1.416116e-02
##  [625,] 0.386693968 0.3625255950 1.132892e-01
##  [626,] 0.211473264 0.4440938538 3.108657e-01
##  [627,] 0.340371253 0.0654560102 4.195898e-03
##  [628,] 0.360146521 0.0776786613 5.584740e-03
##  [629,] 0.406028666 0.1184250277 1.151354e-02
##  [630,] 0.417093250 0.1331148669 1.416116e-02
##  [631,] 0.443086838 0.2436977611 4.467792e-02
##  [632,] 0.436239133 0.2755194522 5.800410e-02
##  [633,] 0.444358195 0.2275981001 3.885821e-02
##  [634,] 0.424154946 0.3063341278 7.374710e-02
##  [635,] 0.430813836 0.2910904300 6.556091e-02
##  [636,] 0.424154946 0.3063341278 7.374710e-02
##  [637,] 0.360146521 0.0776786613 5.584740e-03
##  [638,] 0.397531973 0.3493462791 1.023338e-01
##  [639,] 0.407438488 0.3355375785 9.210835e-02
##  [640,] 0.335537578 0.4074384881 1.649156e-01
##  [641,] 0.444093854 0.2114732637 3.356718e-02
##  [642,] 0.436239133 0.2755194522 5.800410e-02
##  [643,] 0.275519452 0.4362391326 2.302373e-01
##  [644,] 0.360146521 0.0776786613 5.584740e-03
##  [645,] 0.417093250 0.1331148669 1.416116e-02
##  [646,] 0.417093250 0.1331148669 1.416116e-02
##  [647,] 0.440355309 0.2596967205 5.105149e-02
##  [648,] 0.424154946 0.3063341278 7.374710e-02
##  [649,] 0.416337988 0.3211750193 8.258786e-02
##  [650,] 0.243697761 0.4430868383 2.685375e-01
##  [651,] 0.360146521 0.0776786613 5.584740e-03
##  [652,] 0.436239133 0.2755194522 5.800410e-02
##  [653,] 0.397531973 0.3493462791 1.023338e-01
##  [654,] 0.377630828 0.0906313987 7.250512e-03
##  [655,] 0.444358195 0.2275981001 3.885821e-02
##  [656,] 0.375000000 0.3750000000 1.250000e-01
##  [657,] 0.424154946 0.3063341278 7.374710e-02
##  [658,] 0.306334128 0.4241549461 1.957638e-01
##  [659,] 0.436239133 0.2755194522 5.800410e-02
##  [660,] 0.444358195 0.2275981001 3.885821e-02
##  [661,] 0.377630828 0.0906313987 7.250512e-03
##  [662,] 0.417093250 0.1331148669 1.416116e-02
##  [663,] 0.444093854 0.2114732637 3.356718e-02
##  [664,] 0.335537578 0.4074384881 1.649156e-01
##  [665,] 0.306334128 0.4241549461 1.957638e-01
##  [666,] 0.179450170 0.4386559699 3.574234e-01
##  [667,] 0.259696720 0.4403553087 2.488965e-01
##  [668,] 0.406028666 0.1184250277 1.151354e-02
##  [669,] 0.443086838 0.2436977611 4.467792e-02
##  [670,] 0.375000000 0.3750000000 1.250000e-01
##  [671,] 0.306334128 0.4241549461 1.957638e-01
##  [672,] 0.386693968 0.3625255950 1.132892e-01
##  [673,] 0.407438488 0.3355375785 9.210835e-02
##  [674,] 0.377630828 0.0906313987 7.250512e-03
##  [675,] 0.318229499 0.0540389715 3.058810e-03
##  [676,] 0.291090430 0.4308138364 2.125348e-01
##  [677,] 0.406028666 0.1184250277 1.151354e-02
##  [678,] 0.375000000 0.3750000000 1.250000e-01
##  [679,] 0.362525595 0.3866939680 1.374912e-01
##  [680,] 0.362525595 0.3866939680 1.374912e-01
##  [681,] 0.424154946 0.3063341278 7.374710e-02
##  [682,] 0.259696720 0.4403553087 2.488965e-01
##  [683,] 0.043503071 0.2936457319 6.607029e-01
##  [684,] 0.204487093 0.0179374643 5.244873e-04
##  [685,] 0.392899701 0.1042386963 9.218388e-03
##  [686,] 0.407438488 0.3355375785 9.210835e-02
##  [687,] 0.291090430 0.4308138364 2.125348e-01
##  [688,] 0.424154946 0.3063341278 7.374710e-02
##  [689,] 0.424154946 0.3063341278 7.374710e-02
##  [690,] 0.406028666 0.1184250277 1.151354e-02
##  [691,] 0.211473264 0.4440938538 3.108657e-01
##  [692,] 0.386693968 0.3625255950 1.132892e-01
##  [693,] 0.306334128 0.4241549461 1.957638e-01
##  [694,] 0.360146521 0.0776786613 5.584740e-03
##  [695,] 0.433331375 0.1637029640 2.061445e-02
##  [696,] 0.266544426 0.0339238361 1.439193e-03
##  [697,] 0.349346279 0.3975319727 1.507880e-01
##  [698,] 0.417093250 0.1331148669 1.416116e-02
##  [699,] 0.227598100 0.4443581954 2.891855e-01
##  [700,] 0.179450170 0.4386559699 3.574234e-01
##  [701,] 0.340371253 0.0654560102 4.195898e-03
##  [702,] 0.335537578 0.4074384881 1.649156e-01
##  [703,] 0.360146521 0.0776786613 5.584740e-03
##  [704,] 0.426168977 0.1482326877 1.718640e-02
##  [705,] 0.266544426 0.0339238361 1.439193e-03
##  [706,] 0.118425028 0.4060286664 4.640328e-01
##  [707,] 0.430813836 0.2910904300 6.556091e-02
##  [708,] 0.416337988 0.3211750193 8.258786e-02
##  [709,] 0.433331375 0.1637029640 2.061445e-02
##  [710,] 0.375000000 0.3750000000 1.250000e-01
##  [711,] 0.211473264 0.4440938538 3.108657e-01
##  [712,] 0.291090430 0.4308138364 2.125348e-01
##  [713,] 0.406028666 0.1184250277 1.151354e-02
##  [714,] 0.321175019 0.4163379880 1.798991e-01
##  [715,] 0.259696720 0.4403553087 2.488965e-01
##  [716,] 0.349346279 0.3975319727 1.507880e-01
##  [717,] 0.275519452 0.4362391326 2.302373e-01
##  [718,] 0.377630828 0.0906313987 7.250512e-03
##  [719,] 0.131453291 0.0066840657 1.132892e-04
##  [720,] 0.211473264 0.4440938538 3.108657e-01
##  [721,] 0.211473264 0.4440938538 3.108657e-01
##  [722,] 0.386693968 0.3625255950 1.132892e-01
##  [723,] 0.444358195 0.2275981001 3.885821e-02
##  [724,] 0.406028666 0.1184250277 1.151354e-02
##  [725,] 0.349346279 0.3975319727 1.507880e-01
##  [726,] 0.424154946 0.3063341278 7.374710e-02
##  [727,] 0.407438488 0.3355375785 9.210835e-02
##  [728,] 0.236850055 0.0253767916 9.063140e-04
##  [729,] 0.442218287 0.1953987782 2.877966e-02
##  [730,] 0.043503071 0.2936457319 6.607029e-01
##  [731,] 0.362525595 0.3866939680 1.374912e-01
##  [732,] 0.318229499 0.0540389715 3.058810e-03
##  [733,] 0.440355309 0.2596967205 5.105149e-02
##  [734,] 0.090631399 0.0030210466 3.356718e-05
##  [735,] 0.375000000 0.3750000000 1.250000e-01
##  [736,] 0.266544426 0.0339238361 1.439193e-03
##  [737,] 0.321175019 0.4163379880 1.798991e-01
##  [738,] 0.416337988 0.3211750193 8.258786e-02
##  [739,] 0.406028666 0.1184250277 1.151354e-02
##  [740,] 0.397531973 0.3493462791 1.023338e-01
##  [741,] 0.293645732 0.0435030714 2.148300e-03
##  [742,] 0.392899701 0.1042386963 9.218388e-03
##  [743,] 0.406028666 0.1184250277 1.151354e-02
##  [744,] 0.362525595 0.3866939680 1.374912e-01
##  [745,] 0.375000000 0.3750000000 1.250000e-01
##  [746,] 0.266544426 0.0339238361 1.439193e-03
##  [747,] 0.211473264 0.4440938538 3.108657e-01
##  [748,] 0.179450170 0.4386559699 3.574234e-01
##  [749,] 0.163702964 0.4333313752 3.823512e-01
##  [750,] 0.360146521 0.0776786613 5.584740e-03
##  [751,] 0.349346279 0.3975319727 1.507880e-01
##  [752,] 0.340371253 0.0654560102 4.195898e-03
##  [753,] 0.438655970 0.1794501695 2.447048e-02
##  [754,] 0.340371253 0.0654560102 4.195898e-03
##  [755,] 0.444093854 0.2114732637 3.356718e-02
##  [756,] 0.433331375 0.1637029640 2.061445e-02
##  [757,] 0.407438488 0.3355375785 9.210835e-02
##  [758,] 0.442218287 0.1953987782 2.877966e-02
##  [759,] 0.227598100 0.4443581954 2.891855e-01
##  [760,] 0.349346279 0.3975319727 1.507880e-01
##  [761,] 0.293645732 0.0435030714 2.148300e-03
##  [762,] 0.406028666 0.1184250277 1.151354e-02
##  [763,] 0.204487093 0.0179374643 5.244873e-04
##  [764,] 0.362525595 0.3866939680 1.374912e-01
##  [765,] 0.266544426 0.0339238361 1.439193e-03
##  [766,] 0.430813836 0.2910904300 6.556091e-02
##  [767,] 0.438655970 0.1794501695 2.447048e-02
##  [768,] 0.362525595 0.3866939680 1.374912e-01
##  [769,] 0.426168977 0.1482326877 1.718640e-02
##  [770,] 0.426168977 0.1482326877 1.718640e-02
##  [771,] 0.444358195 0.2275981001 3.885821e-02
##  [772,] 0.443086838 0.2436977611 4.467792e-02
##  [773,] 0.406028666 0.1184250277 1.151354e-02
##  [774,] 0.163702964 0.4333313752 3.823512e-01
##  [775,] 0.104238696 0.3928997013 4.936432e-01
##  [776,] 0.444358195 0.2275981001 3.885821e-02
##  [777,] 0.392899701 0.1042386963 9.218388e-03
##  [778,] 0.195398778 0.4422182874 3.336033e-01
##  [779,] 0.131453291 0.0066840657 1.132892e-04
##  [780,] 0.321175019 0.4163379880 1.798991e-01
##  [781,] 0.436239133 0.2755194522 5.800410e-02
##  [782,] 0.306334128 0.4241549461 1.957638e-01
##  [783,] 0.438655970 0.1794501695 2.447048e-02
##  [784,] 0.211473264 0.4440938538 3.108657e-01
##  [785,] 0.436239133 0.2755194522 5.800410e-02
##  [786,] 0.440355309 0.2596967205 5.105149e-02
##  [787,] 0.426168977 0.1482326877 1.718640e-02
##  [788,] 0.169380014 0.0116813803 2.685375e-04
##  [789,] 0.397531973 0.3493462791 1.023338e-01
##  [790,] 0.227598100 0.4443581954 2.891855e-01
##  [791,] 0.360146521 0.0776786613 5.584740e-03
##  [792,] 0.406028666 0.1184250277 1.151354e-02
##  [793,] 0.375000000 0.3750000000 1.250000e-01
##  [794,] 0.417093250 0.1331148669 1.416116e-02
##  [795,] 0.349346279 0.3975319727 1.507880e-01
##  [796,] 0.442218287 0.1953987782 2.877966e-02
##  [797,] 0.163702964 0.4333313752 3.823512e-01
##  [798,] 0.443086838 0.2436977611 4.467792e-02
##  [799,] 0.416337988 0.3211750193 8.258786e-02
##  [800,] 0.133114867 0.4170932496 4.356307e-01
##  [801,] 0.362525595 0.3866939680 1.374912e-01
##  [802,] 0.386693968 0.3625255950 1.132892e-01
##  [803,] 0.377630828 0.0906313987 7.250512e-03
##  [804,] 0.442218287 0.1953987782 2.877966e-02
##  [805,] 0.349346279 0.3975319727 1.507880e-01
##  [806,] 0.291090430 0.4308138364 2.125348e-01
##  [807,] 0.417093250 0.1331148669 1.416116e-02
##  [808,] 0.426168977 0.1482326877 1.718640e-02
##  [809,] 0.375000000 0.3750000000 1.250000e-01
##  [810,] 0.179450170 0.4386559699 3.574234e-01
##  [811,] 0.392899701 0.1042386963 9.218388e-03
##  [812,] 0.430813836 0.2910904300 6.556091e-02
##  [813,] 0.430813836 0.2910904300 6.556091e-02
##  [814,] 0.386693968 0.3625255950 1.132892e-01
##  [815,] 0.386693968 0.3625255950 1.132892e-01
##  [816,] 0.360146521 0.0776786613 5.584740e-03
##  [817,] 0.335537578 0.4074384881 1.649156e-01
##  [818,] 0.443086838 0.2436977611 4.467792e-02
##  [819,] 0.306334128 0.4241549461 1.957638e-01
##  [820,] 0.444093854 0.2114732637 3.356718e-02
##  [821,] 0.340371253 0.0654560102 4.195898e-03
##  [822,] 0.417093250 0.1331148669 1.416116e-02
##  [823,] 0.424154946 0.3063341278 7.374710e-02
##  [824,] 0.440355309 0.2596967205 5.105149e-02
##  [825,] 0.392899701 0.1042386963 9.218388e-03
##  [826,] 0.236850055 0.0253767916 9.063140e-04
##  [827,] 0.426168977 0.1482326877 1.718640e-02
##  [828,] 0.340371253 0.0654560102 4.195898e-03
##  [829,] 0.377630828 0.0906313987 7.250512e-03
##  [830,] 0.416337988 0.3211750193 8.258786e-02
##  [831,] 0.433331375 0.1637029640 2.061445e-02
##  [832,] 0.397531973 0.3493462791 1.023338e-01
##  [833,] 0.054038972 0.3182294988 6.246727e-01
##  [834,] 0.444358195 0.2275981001 3.885821e-02
##  [835,] 0.440355309 0.2596967205 5.105149e-02
##  [836,] 0.090631399 0.0030210466 3.356718e-05
##  [837,] 0.426168977 0.1482326877 1.718640e-02
##  [838,] 0.293645732 0.0435030714 2.148300e-03
##  [839,] 0.349346279 0.3975319727 1.507880e-01
##  [840,] 0.266544426 0.0339238361 1.439193e-03
##  [841,] 0.442218287 0.1953987782 2.877966e-02
##  [842,] 0.291090430 0.4308138364 2.125348e-01
##  [843,] 0.444358195 0.2275981001 3.885821e-02
##  [844,] 0.407438488 0.3355375785 9.210835e-02
##  [845,] 0.386693968 0.3625255950 1.132892e-01
##  [846,] 0.306334128 0.4241549461 1.957638e-01
##  [847,] 0.386693968 0.3625255950 1.132892e-01
##  [848,] 0.397531973 0.3493462791 1.023338e-01
##  [849,] 0.090631399 0.0030210466 3.356718e-05
##  [850,] 0.442218287 0.1953987782 2.877966e-02
##  [851,] 0.407438488 0.3355375785 9.210835e-02
##  [852,] 0.306334128 0.4241549461 1.957638e-01
##  [853,] 0.349346279 0.3975319727 1.507880e-01
##  [854,] 0.406028666 0.1184250277 1.151354e-02
##  [855,] 0.433331375 0.1637029640 2.061445e-02
##  [856,] 0.179450170 0.4386559699 3.574234e-01
##  [857,] 0.397531973 0.3493462791 1.023338e-01
##  [858,] 0.340371253 0.0654560102 4.195898e-03
##  [859,] 0.195398778 0.4422182874 3.336033e-01
##  [860,] 0.293645732 0.0435030714 2.148300e-03
##  [861,] 0.436239133 0.2755194522 5.800410e-02
##  [862,] 0.392899701 0.1042386963 9.218388e-03
##  [863,] 0.424154946 0.3063341278 7.374710e-02
##  [864,] 0.407438488 0.3355375785 9.210835e-02
##  [865,] 0.306334128 0.4241549461 1.957638e-01
##  [866,] 0.443086838 0.2436977611 4.467792e-02
##  [867,] 0.444093854 0.2114732637 3.356718e-02
##  [868,] 0.430813836 0.2910904300 6.556091e-02
##  [869,] 0.377630828 0.0906313987 7.250512e-03
##  [870,] 0.243697761 0.4430868383 2.685375e-01
##  [871,] 0.416337988 0.3211750193 8.258786e-02
##  [872,] 0.397531973 0.3493462791 1.023338e-01
##  [873,] 0.397531973 0.3493462791 1.023338e-01
##  [874,] 0.227598100 0.4443581954 2.891855e-01
##  [875,] 0.443086838 0.2436977611 4.467792e-02
##  [876,] 0.436239133 0.2755194522 5.800410e-02
##  [877,] 0.360146521 0.0776786613 5.584740e-03
##  [878,] 0.243697761 0.4430868383 2.685375e-01
##  [879,] 0.433331375 0.1637029640 2.061445e-02
##  [880,] 0.386693968 0.3625255950 1.132892e-01
##  [881,] 0.318229499 0.0540389715 3.058810e-03
##  [882,] 0.443086838 0.2436977611 4.467792e-02
##  [883,] 0.426168977 0.1482326877 1.718640e-02
##  [884,] 0.090631399 0.0030210466 3.356718e-05
##  [885,] 0.362525595 0.3866939680 1.374912e-01
##  [886,] 0.436239133 0.2755194522 5.800410e-02
##  [887,] 0.416337988 0.3211750193 8.258786e-02
##  [888,] 0.227598100 0.4443581954 2.891855e-01
##  [889,] 0.104238696 0.3928997013 4.936432e-01
##  [890,] 0.293645732 0.0435030714 2.148300e-03
##  [891,] 0.426168977 0.1482326877 1.718640e-02
##  [892,] 0.424154946 0.3063341278 7.374710e-02
##  [893,] 0.321175019 0.4163379880 1.798991e-01
##  [894,] 0.306334128 0.4241549461 1.957638e-01
##  [895,] 0.291090430 0.4308138364 2.125348e-01
##  [896,] 0.377630828 0.0906313987 7.250512e-03
##  [897,] 0.386693968 0.3625255950 1.132892e-01
##  [898,] 0.386693968 0.3625255950 1.132892e-01
##  [899,] 0.377630828 0.0906313987 7.250512e-03
##  [900,] 0.266544426 0.0339238361 1.439193e-03
##  [901,] 0.227598100 0.4443581954 2.891855e-01
##  [902,] 0.444093854 0.2114732637 3.356718e-02
##  [903,] 0.443086838 0.2436977611 4.467792e-02
##  [904,] 0.438655970 0.1794501695 2.447048e-02
##  [905,] 0.340371253 0.0654560102 4.195898e-03
##  [906,] 0.426168977 0.1482326877 1.718640e-02
##  [907,] 0.444358195 0.2275981001 3.885821e-02
##  [908,] 0.340371253 0.0654560102 4.195898e-03
##  [909,] 0.318229499 0.0540389715 3.058810e-03
##  [910,] 0.426168977 0.1482326877 1.718640e-02
##  [911,] 0.444093854 0.2114732637 3.356718e-02
##  [912,] 0.349346279 0.3975319727 1.507880e-01
##  [913,] 0.436239133 0.2755194522 5.800410e-02
##  [914,] 0.406028666 0.1184250277 1.151354e-02
##  [915,] 0.318229499 0.0540389715 3.058810e-03
##  [916,] 0.349346279 0.3975319727 1.507880e-01
##  [917,] 0.266544426 0.0339238361 1.439193e-03
##  [918,] 0.211473264 0.4440938538 3.108657e-01
##  [919,] 0.179450170 0.4386559699 3.574234e-01
##  [920,] 0.321175019 0.4163379880 1.798991e-01
##  [921,] 0.444358195 0.2275981001 3.885821e-02
##  [922,] 0.204487093 0.0179374643 5.244873e-04
##  [923,] 0.397531973 0.3493462791 1.023338e-01
##  [924,] 0.406028666 0.1184250277 1.151354e-02
##  [925,] 0.259696720 0.4403553087 2.488965e-01
##  [926,] 0.243697761 0.4430868383 2.685375e-01
##  [927,] 0.397531973 0.3493462791 1.023338e-01
##  [928,] 0.440355309 0.2596967205 5.105149e-02
##  [929,] 0.318229499 0.0540389715 3.058810e-03
##  [930,] 0.046838810 0.0007678494 4.195898e-06
##  [931,] 0.424154946 0.3063341278 7.374710e-02
##  [932,] 0.406028666 0.1184250277 1.151354e-02
##  [933,] 0.392899701 0.1042386963 9.218388e-03
##  [934,] 0.362525595 0.3866939680 1.374912e-01
##  [935,] 0.335537578 0.4074384881 1.649156e-01
##  [936,] 0.417093250 0.1331148669 1.416116e-02
##  [937,] 0.360146521 0.0776786613 5.584740e-03
##  [938,] 0.426168977 0.1482326877 1.718640e-02
##  [939,] 0.169380014 0.0116813803 2.685375e-04
##  [940,] 0.436239133 0.2755194522 5.800410e-02
##  [941,] 0.424154946 0.3063341278 7.374710e-02
##  [942,] 0.416337988 0.3211750193 8.258786e-02
##  [943,] 0.407438488 0.3355375785 9.210835e-02
##  [944,] 0.227598100 0.4443581954 2.891855e-01
##  [945,] 0.335537578 0.4074384881 1.649156e-01
##  [946,] 0.416337988 0.3211750193 8.258786e-02
##  [947,] 0.321175019 0.4163379880 1.798991e-01
##  [948,] 0.340371253 0.0654560102 4.195898e-03
##  [949,] 0.335537578 0.4074384881 1.649156e-01
##  [950,] 0.440355309 0.2596967205 5.105149e-02
##  [951,] 0.424154946 0.3063341278 7.374710e-02
##  [952,] 0.386693968 0.3625255950 1.132892e-01
##  [953,] 0.397531973 0.3493462791 1.023338e-01
##  [954,] 0.392899701 0.1042386963 9.218388e-03
##  [955,] 0.340371253 0.0654560102 4.195898e-03
##  [956,] 0.416337988 0.3211750193 8.258786e-02
##  [957,] 0.275519452 0.4362391326 2.302373e-01
##  [958,] 0.397531973 0.3493462791 1.023338e-01
##  [959,] 0.440355309 0.2596967205 5.105149e-02
##  [960,] 0.375000000 0.3750000000 1.250000e-01
##  [961,] 0.386693968 0.3625255950 1.132892e-01
##  [962,] 0.259696720 0.4403553087 2.488965e-01
##  [963,] 0.416337988 0.3211750193 8.258786e-02
##  [964,] 0.335537578 0.4074384881 1.649156e-01
##  [965,] 0.349346279 0.3975319727 1.507880e-01
##  [966,] 0.407438488 0.3355375785 9.210835e-02
##  [967,] 0.416337988 0.3211750193 8.258786e-02
##  [968,] 0.443086838 0.2436977611 4.467792e-02
##  [969,] 0.386693968 0.3625255950 1.132892e-01
##  [970,] 0.397531973 0.3493462791 1.023338e-01
##  [971,] 0.416337988 0.3211750193 8.258786e-02
##  [972,] 0.375000000 0.3750000000 1.250000e-01
##  [973,] 0.259696720 0.4403553087 2.488965e-01
##  [974,] 0.006684066 0.1314532913 8.617494e-01
##  [975,] 0.386693968 0.3625255950 1.132892e-01
##  [976,] 0.275519452 0.4362391326 2.302373e-01
##  [977,] 0.444358195 0.2275981001 3.885821e-02
##  [978,] 0.424154946 0.3063341278 7.374710e-02
##  [979,] 0.375000000 0.3750000000 1.250000e-01
##  [980,] 0.243697761 0.4430868383 2.685375e-01
##  [981,] 0.407438488 0.3355375785 9.210835e-02
##  [982,] 0.293645732 0.0435030714 2.148300e-03
##  [983,] 0.195398778 0.4422182874 3.336033e-01
##  [984,] 0.179450170 0.4386559699 3.574234e-01
##  [985,] 0.397531973 0.3493462791 1.023338e-01
##  [986,] 0.443086838 0.2436977611 4.467792e-02
##  [987,] 0.433331375 0.1637029640 2.061445e-02
##  [988,] 0.195398778 0.4422182874 3.336033e-01
##  [989,] 0.416337988 0.3211750193 8.258786e-02
##  [990,] 0.318229499 0.0540389715 3.058810e-03
##  [991,] 0.360146521 0.0776786613 5.584740e-03
##  [992,] 0.362525595 0.3866939680 1.374912e-01
##  [993,] 0.266544426 0.0339238361 1.439193e-03
##  [994,] 0.440355309 0.2596967205 5.105149e-02
##  [995,] 0.444093854 0.2114732637 3.356718e-02
##  [996,] 0.438655970 0.1794501695 2.447048e-02
##  [997,] 0.204487093 0.0179374643 5.244873e-04
##  [998,] 0.340371253 0.0654560102 4.195898e-03
##  [999,] 0.436239133 0.2755194522 5.800410e-02
## [1000,] 0.442218287 0.1953987782 2.877966e-02
## [1001,] 0.243697761 0.4430868383 2.685375e-01
## [1002,] 0.148232688 0.4261689772 4.084119e-01
## [1003,] 0.416337988 0.3211750193 8.258786e-02
## [1004,] 0.443086838 0.2436977611 4.467792e-02
## [1005,] 0.291090430 0.4308138364 2.125348e-01
## [1006,] 0.407438488 0.3355375785 9.210835e-02
## [1007,] 0.291090430 0.4308138364 2.125348e-01
## [1008,] 0.321175019 0.4163379880 1.798991e-01
## [1009,] 0.417093250 0.1331148669 1.416116e-02
## [1010,] 0.306334128 0.4241549461 1.957638e-01
## [1011,] 0.406028666 0.1184250277 1.151354e-02
## [1012,] 0.306334128 0.4241549461 1.957638e-01
## [1013,] 0.444093854 0.2114732637 3.356718e-02
## [1014,] 0.392899701 0.1042386963 9.218388e-03
## [1015,] 0.440355309 0.2596967205 5.105149e-02
## [1016,] 0.416337988 0.3211750193 8.258786e-02
## [1017,] 0.375000000 0.3750000000 1.250000e-01
## [1018,] 0.362525595 0.3866939680 1.374912e-01
## [1019,] 0.443086838 0.2436977611 4.467792e-02
## [1020,] 0.360146521 0.0776786613 5.584740e-03
## [1021,] 0.406028666 0.1184250277 1.151354e-02
## [1022,] 0.349346279 0.3975319727 1.507880e-01
## [1023,] 0.436239133 0.2755194522 5.800410e-02
## [1024,] 0.227598100 0.4443581954 2.891855e-01
## [1025,] 0.392899701 0.1042386963 9.218388e-03
## [1026,] 0.360146521 0.0776786613 5.584740e-03
## [1027,] 0.293645732 0.0435030714 2.148300e-03
## [1028,] 0.362525595 0.3866939680 1.374912e-01
## [1029,] 0.179450170 0.4386559699 3.574234e-01
## [1030,] 0.433331375 0.1637029640 2.061445e-02
## [1031,] 0.169380014 0.0116813803 2.685375e-04
## [1032,] 0.291090430 0.4308138364 2.125348e-01
## [1033,] 0.163702964 0.4333313752 3.823512e-01
## [1034,] 0.430813836 0.2910904300 6.556091e-02
## [1035,] 0.375000000 0.3750000000 1.250000e-01
## [1036,] 0.438655970 0.1794501695 2.447048e-02
## [1037,] 0.293645732 0.0435030714 2.148300e-03
## [1038,] 0.407438488 0.3355375785 9.210835e-02
## [1039,] 0.169380014 0.0116813803 2.685375e-04
## [1040,] 0.163702964 0.4333313752 3.823512e-01
## [1041,] 0.424154946 0.3063341278 7.374710e-02
## [1042,] 0.349346279 0.3975319727 1.507880e-01
## [1043,] 0.407438488 0.3355375785 9.210835e-02
## [1044,] 0.430813836 0.2910904300 6.556091e-02
## [1045,] 0.443086838 0.2436977611 4.467792e-02
## [1046,] 0.440355309 0.2596967205 5.105149e-02
## [1047,] 0.349346279 0.3975319727 1.507880e-01
## [1048,] 0.426168977 0.1482326877 1.718640e-02
## [1049,] 0.416337988 0.3211750193 8.258786e-02
## [1050,] 0.433331375 0.1637029640 2.061445e-02
## [1051,] 0.417093250 0.1331148669 1.416116e-02
## [1052,] 0.407438488 0.3355375785 9.210835e-02
## [1053,] 0.424154946 0.3063341278 7.374710e-02
## [1054,] 0.362525595 0.3866939680 1.374912e-01
## [1055,] 0.291090430 0.4308138364 2.125348e-01
## [1056,] 0.375000000 0.3750000000 1.250000e-01
## [1057,] 0.397531973 0.3493462791 1.023338e-01
## [1058,] 0.443086838 0.2436977611 4.467792e-02
## [1059,] 0.131453291 0.0066840657 1.132892e-04
## [1060,] 0.211473264 0.4440938538 3.108657e-01
## [1061,] 0.275519452 0.4362391326 2.302373e-01
## [1062,] 0.195398778 0.4422182874 3.336033e-01
## [1063,] 0.424154946 0.3063341278 7.374710e-02
## [1064,] 0.430813836 0.2910904300 6.556091e-02
## [1065,] 0.360146521 0.0776786613 5.584740e-03
## [1066,] 0.444093854 0.2114732637 3.356718e-02
## [1067,] 0.293645732 0.0435030714 2.148300e-03
## [1068,] 0.340371253 0.0654560102 4.195898e-03
## [1069,] 0.416337988 0.3211750193 8.258786e-02
## [1070,] 0.444358195 0.2275981001 3.885821e-02
## [1071,] 0.417093250 0.1331148669 1.416116e-02
## [1072,] 0.424154946 0.3063341278 7.374710e-02
## [1073,] 0.386693968 0.3625255950 1.132892e-01
## [1074,] 0.416337988 0.3211750193 8.258786e-02
## [1075,] 0.275519452 0.4362391326 2.302373e-01
## [1076,] 0.443086838 0.2436977611 4.467792e-02
## [1077,] 0.054038972 0.3182294988 6.246727e-01
## [1078,] 0.377630828 0.0906313987 7.250512e-03
## [1079,] 0.416337988 0.3211750193 8.258786e-02
## [1080,] 0.440355309 0.2596967205 5.105149e-02
## [1081,] 0.443086838 0.2436977611 4.467792e-02
## [1082,] 0.227598100 0.4443581954 2.891855e-01
## [1083,] 0.444093854 0.2114732637 3.356718e-02
## [1084,] 0.293645732 0.0435030714 2.148300e-03
## [1085,] 0.321175019 0.4163379880 1.798991e-01
## [1086,] 0.407438488 0.3355375785 9.210835e-02
## [1087,] 0.436239133 0.2755194522 5.800410e-02
## [1088,] 0.377630828 0.0906313987 7.250512e-03
## [1089,] 0.426168977 0.1482326877 1.718640e-02
## [1090,] 0.335537578 0.4074384881 1.649156e-01
## [1091,] 0.335537578 0.4074384881 1.649156e-01
## [1092,] 0.306334128 0.4241549461 1.957638e-01
## [1093,] 0.397531973 0.3493462791 1.023338e-01
## [1094,] 0.131453291 0.0066840657 1.132892e-04
## [1095,] 0.043503071 0.2936457319 6.607029e-01
## [1096,] 0.444093854 0.2114732637 3.356718e-02
## [1097,] 0.321175019 0.4163379880 1.798991e-01
## [1098,] 0.433331375 0.1637029640 2.061445e-02
## [1099,] 0.211473264 0.4440938538 3.108657e-01
## [1100,] 0.444358195 0.2275981001 3.885821e-02
## [1101,] 0.195398778 0.4422182874 3.336033e-01
## [1102,] 0.148232688 0.4261689772 4.084119e-01
## [1103,] 0.407438488 0.3355375785 9.210835e-02
## [1104,] 0.266544426 0.0339238361 1.439193e-03
## [1105,] 0.000000000 0.0000000000 1.000000e+00
## [1106,] 0.349346279 0.3975319727 1.507880e-01
## [1107,] 0.243697761 0.4430868383 2.685375e-01
## [1108,] 0.335537578 0.4074384881 1.649156e-01
## [1109,] 0.416337988 0.3211750193 8.258786e-02
## [1110,] 0.392899701 0.1042386963 9.218388e-03
## [1111,] 0.375000000 0.3750000000 1.250000e-01
## [1112,] 0.397531973 0.3493462791 1.023338e-01
## [1113,] 0.444358195 0.2275981001 3.885821e-02
## [1114,] 0.321175019 0.4163379880 1.798991e-01
## [1115,] 0.442218287 0.1953987782 2.877966e-02
## [1116,] 0.335537578 0.4074384881 1.649156e-01
## [1117,] 0.444358195 0.2275981001 3.885821e-02
## [1118,] 0.163702964 0.4333313752 3.823512e-01
## [1119,] 0.204487093 0.0179374643 5.244873e-04
## [1120,] 0.179450170 0.4386559699 3.574234e-01
## [1121,] 0.430813836 0.2910904300 6.556091e-02
## [1122,] 0.426168977 0.1482326877 1.718640e-02
## [1123,] 0.444093854 0.2114732637 3.356718e-02
## [1124,] 0.266544426 0.0339238361 1.439193e-03
## [1125,] 0.377630828 0.0906313987 7.250512e-03
## [1126,] 0.417093250 0.1331148669 1.416116e-02
## [1127,] 0.360146521 0.0776786613 5.584740e-03
## [1128,] 0.406028666 0.1184250277 1.151354e-02
## [1129,] 0.306334128 0.4241549461 1.957638e-01
## [1130,] 0.236850055 0.0253767916 9.063140e-04
## [1131,] 0.377630828 0.0906313987 7.250512e-03
## [1132,] 0.397531973 0.3493462791 1.023338e-01
## [1133,] 0.424154946 0.3063341278 7.374710e-02
## [1134,] 0.440355309 0.2596967205 5.105149e-02
## [1135,] 0.306334128 0.4241549461 1.957638e-01
## [1136,] 0.266544426 0.0339238361 1.439193e-03
## [1137,] 0.375000000 0.3750000000 1.250000e-01
## [1138,] 0.433331375 0.1637029640 2.061445e-02
## [1139,] 0.118425028 0.4060286664 4.640328e-01
## [1140,] 0.259696720 0.4403553087 2.488965e-01
## [1141,] 0.397531973 0.3493462791 1.023338e-01
## [1142,] 0.275519452 0.4362391326 2.302373e-01
## [1143,] 0.426168977 0.1482326877 1.718640e-02
## [1144,] 0.204487093 0.0179374643 5.244873e-04
## [1145,] 0.430813836 0.2910904300 6.556091e-02
## [1146,] 0.438655970 0.1794501695 2.447048e-02
## [1147,] 0.169380014 0.0116813803 2.685375e-04
## [1148,] 0.362525595 0.3866939680 1.374912e-01
## [1149,] 0.243697761 0.4430868383 2.685375e-01
## [1150,] 0.424154946 0.3063341278 7.374710e-02
## [1151,] 0.362525595 0.3866939680 1.374912e-01
## [1152,] 0.291090430 0.4308138364 2.125348e-01
## [1153,] 0.406028666 0.1184250277 1.151354e-02
## [1154,] 0.362525595 0.3866939680 1.374912e-01
## [1155,] 0.236850055 0.0253767916 9.063140e-04
## [1156,] 0.321175019 0.4163379880 1.798991e-01
## [1157,] 0.266544426 0.0339238361 1.439193e-03
## [1158,] 0.259696720 0.4403553087 2.488965e-01
## [1159,] 0.430813836 0.2910904300 6.556091e-02
## [1160,] 0.443086838 0.2436977611 4.467792e-02
## [1161,] 0.444358195 0.2275981001 3.885821e-02
## [1162,] 0.406028666 0.1184250277 1.151354e-02
## [1163,] 0.386693968 0.3625255950 1.132892e-01
## [1164,] 0.433331375 0.1637029640 2.061445e-02
## [1165,] 0.335537578 0.4074384881 1.649156e-01
## [1166,] 0.362525595 0.3866939680 1.374912e-01
## [1167,] 0.433331375 0.1637029640 2.061445e-02
## [1168,] 0.318229499 0.0540389715 3.058810e-03
## [1169,] 0.259696720 0.4403553087 2.488965e-01
## [1170,] 0.386693968 0.3625255950 1.132892e-01
## [1171,] 0.440355309 0.2596967205 5.105149e-02
## [1172,] 0.227598100 0.4443581954 2.891855e-01
## [1173,] 0.291090430 0.4308138364 2.125348e-01
## [1174,] 0.426168977 0.1482326877 1.718640e-02
## [1175,] 0.430813836 0.2910904300 6.556091e-02
## [1176,] 0.430813836 0.2910904300 6.556091e-02
## [1177,] 0.417093250 0.1331148669 1.416116e-02
## [1178,] 0.131453291 0.0066840657 1.132892e-04
## [1179,] 0.306334128 0.4241549461 1.957638e-01
## [1180,] 0.306334128 0.4241549461 1.957638e-01
## [1181,] 0.433331375 0.1637029640 2.061445e-02
## [1182,] 0.204487093 0.0179374643 5.244873e-04
## [1183,] 0.195398778 0.4422182874 3.336033e-01
## [1184,] 0.349346279 0.3975319727 1.507880e-01
## [1185,] 0.090631399 0.0030210466 3.356718e-05
## [1186,] 0.349346279 0.3975319727 1.507880e-01
## [1187,] 0.133114867 0.4170932496 4.356307e-01
## [1188,] 0.442218287 0.1953987782 2.877966e-02
## [1189,] 0.236850055 0.0253767916 9.063140e-04
## [1190,] 0.438655970 0.1794501695 2.447048e-02
## [1191,] 0.417093250 0.1331148669 1.416116e-02
## [1192,] 0.438655970 0.1794501695 2.447048e-02
## [1193,] 0.406028666 0.1184250277 1.151354e-02
## [1194,] 0.416337988 0.3211750193 8.258786e-02
## [1195,] 0.417093250 0.1331148669 1.416116e-02
## [1196,] 0.397531973 0.3493462791 1.023338e-01
## [1197,] 0.442218287 0.1953987782 2.877966e-02
## [1198,] 0.259696720 0.4403553087 2.488965e-01
## [1199,] 0.397531973 0.3493462791 1.023338e-01
## [1200,] 0.360146521 0.0776786613 5.584740e-03
## [1201,] 0.442218287 0.1953987782 2.877966e-02
## [1202,] 0.259696720 0.4403553087 2.488965e-01
## [1203,] 0.444358195 0.2275981001 3.885821e-02
## [1204,] 0.227598100 0.4443581954 2.891855e-01
## [1205,] 0.392899701 0.1042386963 9.218388e-03
## [1206,] 0.293645732 0.0435030714 2.148300e-03
## [1207,] 0.444093854 0.2114732637 3.356718e-02
## [1208,] 0.349346279 0.3975319727 1.507880e-01
## [1209,] 0.406028666 0.1184250277 1.151354e-02
## [1210,] 0.375000000 0.3750000000 1.250000e-01
## [1211,] 0.443086838 0.2436977611 4.467792e-02
## [1212,] 0.211473264 0.4440938538 3.108657e-01
## [1213,] 0.377630828 0.0906313987 7.250512e-03
## [1214,] 0.440355309 0.2596967205 5.105149e-02
## [1215,] 0.406028666 0.1184250277 1.151354e-02
## [1216,] 0.440355309 0.2596967205 5.105149e-02
## [1217,] 0.321175019 0.4163379880 1.798991e-01
## [1218,] 0.433331375 0.1637029640 2.061445e-02
## [1219,] 0.430813836 0.2910904300 6.556091e-02
## [1220,] 0.362525595 0.3866939680 1.374912e-01
## [1221,] 0.046838810 0.0007678494 4.195898e-06
## [1222,] 0.321175019 0.4163379880 1.798991e-01
## [1223,] 0.169380014 0.0116813803 2.685375e-04
## [1224,] 0.375000000 0.3750000000 1.250000e-01
## [1225,] 0.417093250 0.1331148669 1.416116e-02
## [1226,] 0.392899701 0.1042386963 9.218388e-03
## [1227,] 0.430813836 0.2910904300 6.556091e-02
## [1228,] 0.443086838 0.2436977611 4.467792e-02
## [1229,] 0.386693968 0.3625255950 1.132892e-01
## [1230,] 0.407438488 0.3355375785 9.210835e-02
## [1231,] 0.243697761 0.4430868383 2.685375e-01
## [1232,] 0.362525595 0.3866939680 1.374912e-01
## [1233,] 0.444093854 0.2114732637 3.356718e-02
## [1234,] 0.417093250 0.1331148669 1.416116e-02
## [1235,] 0.335537578 0.4074384881 1.649156e-01
## [1236,] 0.321175019 0.4163379880 1.798991e-01
## [1237,] 0.442218287 0.1953987782 2.877966e-02
## [1238,] 0.306334128 0.4241549461 1.957638e-01
## [1239,] 0.306334128 0.4241549461 1.957638e-01
## [1240,] 0.266544426 0.0339238361 1.439193e-03
## [1241,] 0.433331375 0.1637029640 2.061445e-02
## [1242,] 0.360146521 0.0776786613 5.584740e-03
## [1243,] 0.430813836 0.2910904300 6.556091e-02
## [1244,] 0.291090430 0.4308138364 2.125348e-01
## [1245,] 0.386693968 0.3625255950 1.132892e-01
## [1246,] 0.436239133 0.2755194522 5.800410e-02
## [1247,] 0.430813836 0.2910904300 6.556091e-02
## [1248,] 0.406028666 0.1184250277 1.151354e-02
## [1249,] 0.090631399 0.0030210466 3.356718e-05
## [1250,] 0.430813836 0.2910904300 6.556091e-02
## [1251,] 0.243697761 0.4430868383 2.685375e-01
## [1252,] 0.444093854 0.2114732637 3.356718e-02
## [1253,] 0.204487093 0.0179374643 5.244873e-04
## [1254,] 0.306334128 0.4241549461 1.957638e-01
## [1255,] 0.118425028 0.4060286664 4.640328e-01
## [1256,] 0.397531973 0.3493462791 1.023338e-01
## [1257,] 0.444358195 0.2275981001 3.885821e-02
## [1258,] 0.433331375 0.1637029640 2.061445e-02
## [1259,] 0.443086838 0.2436977611 4.467792e-02
## [1260,] 0.443086838 0.2436977611 4.467792e-02
## [1261,] 0.433331375 0.1637029640 2.061445e-02
## [1262,] 0.293645732 0.0435030714 2.148300e-03
## [1263,] 0.204487093 0.0179374643 5.244873e-04
## [1264,] 0.195398778 0.4422182874 3.336033e-01
## [1265,] 0.236850055 0.0253767916 9.063140e-04
## [1266,] 0.362525595 0.3866939680 1.374912e-01
## [1267,] 0.169380014 0.0116813803 2.685375e-04
## [1268,] 0.179450170 0.4386559699 3.574234e-01
## [1269,] 0.440355309 0.2596967205 5.105149e-02
## [1270,] 0.306334128 0.4241549461 1.957638e-01
## [1271,] 0.360146521 0.0776786613 5.584740e-03
## [1272,] 0.444358195 0.2275981001 3.885821e-02
## [1273,] 0.054038972 0.3182294988 6.246727e-01
## [1274,] 0.169380014 0.0116813803 2.685375e-04
## [1275,] 0.386693968 0.3625255950 1.132892e-01
## [1276,] 0.433331375 0.1637029640 2.061445e-02
## [1277,] 0.407438488 0.3355375785 9.210835e-02
## [1278,] 0.291090430 0.4308138364 2.125348e-01
## [1279,] 0.438655970 0.1794501695 2.447048e-02
## [1280,] 0.131453291 0.0066840657 1.132892e-04
## [1281,] 0.440355309 0.2596967205 5.105149e-02
## [1282,] 0.406028666 0.1184250277 1.151354e-02
## [1283,] 0.438655970 0.1794501695 2.447048e-02
## [1284,] 0.340371253 0.0654560102 4.195898e-03
## [1285,] 0.440355309 0.2596967205 5.105149e-02
## [1286,] 0.291090430 0.4308138364 2.125348e-01
## [1287,] 0.424154946 0.3063341278 7.374710e-02
## [1288,] 0.440355309 0.2596967205 5.105149e-02
## [1289,] 0.259696720 0.4403553087 2.488965e-01
## [1290,] 0.291090430 0.4308138364 2.125348e-01
## [1291,] 0.438655970 0.1794501695 2.447048e-02
## [1292,] 0.430813836 0.2910904300 6.556091e-02
## [1293,] 0.318229499 0.0540389715 3.058810e-03
## [1294,] 0.406028666 0.1184250277 1.151354e-02
## [1295,] 0.444093854 0.2114732637 3.356718e-02
## [1296,] 0.340371253 0.0654560102 4.195898e-03
## [1297,] 0.436239133 0.2755194522 5.800410e-02
## [1298,] 0.349346279 0.3975319727 1.507880e-01
## [1299,] 0.291090430 0.4308138364 2.125348e-01
## [1300,] 0.444358195 0.2275981001 3.885821e-02
## [1301,] 0.436239133 0.2755194522 5.800410e-02
## [1302,] 0.204487093 0.0179374643 5.244873e-04
## [1303,] 0.443086838 0.2436977611 4.467792e-02
## [1304,] 0.443086838 0.2436977611 4.467792e-02
## [1305,] 0.349346279 0.3975319727 1.507880e-01
## [1306,] 0.011681380 0.1693800141 8.186701e-01
## [1307,] 0.318229499 0.0540389715 3.058810e-03
## [1308,] 0.266544426 0.0339238361 1.439193e-03
## [1309,] 0.318229499 0.0540389715 3.058810e-03
## [1310,] 0.417093250 0.1331148669 1.416116e-02
## [1311,] 0.349346279 0.3975319727 1.507880e-01
## [1312,] 0.169380014 0.0116813803 2.685375e-04
## [1313,] 0.397531973 0.3493462791 1.023338e-01
## [1314,] 0.426168977 0.1482326877 1.718640e-02
## [1315,] 0.397531973 0.3493462791 1.023338e-01
## [1316,] 0.392899701 0.1042386963 9.218388e-03
## [1317,] 0.397531973 0.3493462791 1.023338e-01
## [1318,] 0.375000000 0.3750000000 1.250000e-01
## [1319,] 0.443086838 0.2436977611 4.467792e-02
## [1320,] 0.349346279 0.3975319727 1.507880e-01
## [1321,] 0.392899701 0.1042386963 9.218388e-03
## [1322,] 0.386693968 0.3625255950 1.132892e-01
## [1323,] 0.275519452 0.4362391326 2.302373e-01
## [1324,] 0.407438488 0.3355375785 9.210835e-02
## [1325,] 0.321175019 0.4163379880 1.798991e-01
## [1326,] 0.406028666 0.1184250277 1.151354e-02
## [1327,] 0.291090430 0.4308138364 2.125348e-01
## [1328,] 0.433331375 0.1637029640 2.061445e-02
## [1329,] 0.417093250 0.1331148669 1.416116e-02
## [1330,] 0.417093250 0.1331148669 1.416116e-02
## [1331,] 0.440355309 0.2596967205 5.105149e-02
## [1332,] 0.436239133 0.2755194522 5.800410e-02
## [1333,] 0.243697761 0.4430868383 2.685375e-01
## [1334,] 0.416337988 0.3211750193 8.258786e-02
## [1335,] 0.397531973 0.3493462791 1.023338e-01
## [1336,] 0.426168977 0.1482326877 1.718640e-02
## [1337,] 0.430813836 0.2910904300 6.556091e-02
## [1338,] 0.243697761 0.4430868383 2.685375e-01
## [1339,] 0.424154946 0.3063341278 7.374710e-02
## [1340,] 0.438655970 0.1794501695 2.447048e-02
## [1341,] 0.397531973 0.3493462791 1.023338e-01
## [1342,] 0.275519452 0.4362391326 2.302373e-01
## [1343,] 0.444093854 0.2114732637 3.356718e-02
## [1344,] 0.424154946 0.3063341278 7.374710e-02
## [1345,] 0.275519452 0.4362391326 2.302373e-01
## [1346,] 0.349346279 0.3975319727 1.507880e-01
## [1347,] 0.440355309 0.2596967205 5.105149e-02
## [1348,] 0.335537578 0.4074384881 1.649156e-01
## [1349,] 0.318229499 0.0540389715 3.058810e-03
## [1350,] 0.335537578 0.4074384881 1.649156e-01
## [1351,] 0.349346279 0.3975319727 1.507880e-01
## [1352,] 0.349346279 0.3975319727 1.507880e-01
## [1353,] 0.340371253 0.0654560102 4.195898e-03
## [1354,] 0.375000000 0.3750000000 1.250000e-01
## [1355,] 0.195398778 0.4422182874 3.336033e-01
## [1356,] 0.204487093 0.0179374643 5.244873e-04
## [1357,] 0.321175019 0.4163379880 1.798991e-01
## [1358,] 0.291090430 0.4308138364 2.125348e-01
## [1359,] 0.386693968 0.3625255950 1.132892e-01
## [1360,] 0.362525595 0.3866939680 1.374912e-01
## [1361,] 0.375000000 0.3750000000 1.250000e-01
## [1362,] 0.375000000 0.3750000000 1.250000e-01
## [1363,] 0.430813836 0.2910904300 6.556091e-02
## [1364,] 0.407438488 0.3355375785 9.210835e-02
## [1365,] 0.386693968 0.3625255950 1.132892e-01
## [1366,] 0.046838810 0.0007678494 4.195898e-06
## [1367,] 0.275519452 0.4362391326 2.302373e-01
## [1368,] 0.424154946 0.3063341278 7.374710e-02
## [1369,] 0.436239133 0.2755194522 5.800410e-02
## [1370,] 0.406028666 0.1184250277 1.151354e-02
## [1371,] 0.406028666 0.1184250277 1.151354e-02
## [1372,] 0.430813836 0.2910904300 6.556091e-02
## [1373,] 0.259696720 0.4403553087 2.488965e-01
## [1374,] 0.104238696 0.3928997013 4.936432e-01
## [1375,] 0.392899701 0.1042386963 9.218388e-03
## [1376,] 0.375000000 0.3750000000 1.250000e-01
## [1377,] 0.440355309 0.2596967205 5.105149e-02
## [1378,] 0.433331375 0.1637029640 2.061445e-02
## [1379,] 0.417093250 0.1331148669 1.416116e-02
## [1380,] 0.321175019 0.4163379880 1.798991e-01
## [1381,] 0.430813836 0.2910904300 6.556091e-02
## [1382,] 0.438655970 0.1794501695 2.447048e-02
## [1383,] 0.444093854 0.2114732637 3.356718e-02
## [1384,] 0.243697761 0.4430868383 2.685375e-01
## [1385,] 0.416337988 0.3211750193 8.258786e-02
## [1386,] 0.426168977 0.1482326877 1.718640e-02
## [1387,] 0.131453291 0.0066840657 1.132892e-04
## [1388,] 0.444358195 0.2275981001 3.885821e-02
## [1389,] 0.340371253 0.0654560102 4.195898e-03
## [1390,] 0.306334128 0.4241549461 1.957638e-01
## [1391,] 0.236850055 0.0253767916 9.063140e-04
## [1392,] 0.392899701 0.1042386963 9.218388e-03
## [1393,] 0.424154946 0.3063341278 7.374710e-02
## [1394,] 0.377630828 0.0906313987 7.250512e-03
## [1395,] 0.440355309 0.2596967205 5.105149e-02
## [1396,] 0.293645732 0.0435030714 2.148300e-03
## [1397,] 0.406028666 0.1184250277 1.151354e-02
## [1398,] 0.436239133 0.2755194522 5.800410e-02
## [1399,] 0.424154946 0.3063341278 7.374710e-02
## [1400,] 0.377630828 0.0906313987 7.250512e-03
## [1401,] 0.243697761 0.4430868383 2.685375e-01
## [1402,] 0.417093250 0.1331148669 1.416116e-02
## [1403,] 0.340371253 0.0654560102 4.195898e-03
## [1404,] 0.430813836 0.2910904300 6.556091e-02
## [1405,] 0.375000000 0.3750000000 1.250000e-01
## [1406,] 0.438655970 0.1794501695 2.447048e-02
## [1407,] 0.397531973 0.3493462791 1.023338e-01
## [1408,] 0.426168977 0.1482326877 1.718640e-02
## [1409,] 0.179450170 0.4386559699 3.574234e-01
## [1410,] 0.424154946 0.3063341278 7.374710e-02
## [1411,] 0.386693968 0.3625255950 1.132892e-01
## [1412,] 0.275519452 0.4362391326 2.302373e-01
## [1413,] 0.362525595 0.3866939680 1.374912e-01
## [1414,] 0.377630828 0.0906313987 7.250512e-03
## [1415,] 0.426168977 0.1482326877 1.718640e-02
## [1416,] 0.349346279 0.3975319727 1.507880e-01
## [1417,] 0.321175019 0.4163379880 1.798991e-01
## [1418,] 0.443086838 0.2436977611 4.467792e-02
## [1419,] 0.426168977 0.1482326877 1.718640e-02
## [1420,] 0.438655970 0.1794501695 2.447048e-02
## [1421,] 0.306334128 0.4241549461 1.957638e-01
## [1422,] 0.179450170 0.4386559699 3.574234e-01
## [1423,] 0.417093250 0.1331148669 1.416116e-02
## [1424,] 0.424154946 0.3063341278 7.374710e-02
## [1425,] 0.000000000 0.0000000000 1.000000e+00
## [1426,] 0.349346279 0.3975319727 1.507880e-01
## [1427,] 0.211473264 0.4440938538 3.108657e-01
## [1428,] 0.417093250 0.1331148669 1.416116e-02
## [1429,] 0.340371253 0.0654560102 4.195898e-03
## [1430,] 0.275519452 0.4362391326 2.302373e-01
## [1431,] 0.275519452 0.4362391326 2.302373e-01
## [1432,] 0.426168977 0.1482326877 1.718640e-02
## [1433,] 0.416337988 0.3211750193 8.258786e-02
## [1434,] 0.275519452 0.4362391326 2.302373e-01
## [1435,] 0.340371253 0.0654560102 4.195898e-03
## [1436,] 0.442218287 0.1953987782 2.877966e-02
## [1437,] 0.275519452 0.4362391326 2.302373e-01
## [1438,] 0.169380014 0.0116813803 2.685375e-04
## [1439,] 0.211473264 0.4440938538 3.108657e-01
## [1440,] 0.377630828 0.0906313987 7.250512e-03
## [1441,] 0.362525595 0.3866939680 1.374912e-01
## [1442,] 0.444093854 0.2114732637 3.356718e-02
## [1443,] 0.291090430 0.4308138364 2.125348e-01
## [1444,] 0.444358195 0.2275981001 3.885821e-02
## [1445,] 0.436239133 0.2755194522 5.800410e-02
## [1446,] 0.054038972 0.3182294988 6.246727e-01
## [1447,] 0.375000000 0.3750000000 1.250000e-01
## [1448,] 0.416337988 0.3211750193 8.258786e-02
## [1449,] 0.440355309 0.2596967205 5.105149e-02
## [1450,] 0.417093250 0.1331148669 1.416116e-02
## [1451,] 0.397531973 0.3493462791 1.023338e-01
## [1452,] 0.204487093 0.0179374643 5.244873e-04
## [1453,] 0.406028666 0.1184250277 1.151354e-02
## [1454,] 0.377630828 0.0906313987 7.250512e-03
## [1455,] 0.306334128 0.4241549461 1.957638e-01
## [1456,] 0.335537578 0.4074384881 1.649156e-01
## [1457,] 0.377630828 0.0906313987 7.250512e-03
## [1458,] 0.406028666 0.1184250277 1.151354e-02
## [1459,] 0.321175019 0.4163379880 1.798991e-01
## [1460,] 0.392899701 0.1042386963 9.218388e-03
## [1461,] 0.362525595 0.3866939680 1.374912e-01
## [1462,] 0.440355309 0.2596967205 5.105149e-02
## [1463,] 0.397531973 0.3493462791 1.023338e-01
## [1464,] 0.442218287 0.1953987782 2.877966e-02
## [1465,] 0.236850055 0.0253767916 9.063140e-04
## [1466,] 0.321175019 0.4163379880 1.798991e-01
## [1467,] 0.444358195 0.2275981001 3.885821e-02
## [1468,] 0.397531973 0.3493462791 1.023338e-01
## [1469,] 0.438655970 0.1794501695 2.447048e-02
## [1470,] 0.211473264 0.4440938538 3.108657e-01
## [1471,] 0.430813836 0.2910904300 6.556091e-02
## [1472,] 0.090631399 0.0030210466 3.356718e-05
## [1473,] 0.318229499 0.0540389715 3.058810e-03
## [1474,] 0.362525595 0.3866939680 1.374912e-01
## [1475,] 0.275519452 0.4362391326 2.302373e-01
## [1476,] 0.046838810 0.0007678494 4.195898e-06
## [1477,] 0.433331375 0.1637029640 2.061445e-02
## [1478,] 0.416337988 0.3211750193 8.258786e-02
## [1479,] 0.306334128 0.4241549461 1.957638e-01
## [1480,] 0.436239133 0.2755194522 5.800410e-02
## [1481,] 0.349346279 0.3975319727 1.507880e-01
## [1482,] 0.386693968 0.3625255950 1.132892e-01
## [1483,] 0.362525595 0.3866939680 1.374912e-01
## [1484,] 0.442218287 0.1953987782 2.877966e-02
## [1485,] 0.444093854 0.2114732637 3.356718e-02
## [1486,] 0.440355309 0.2596967205 5.105149e-02
## [1487,] 0.349346279 0.3975319727 1.507880e-01
## [1488,] 0.349346279 0.3975319727 1.507880e-01
## [1489,] 0.430813836 0.2910904300 6.556091e-02
## [1490,] 0.426168977 0.1482326877 1.718640e-02
## [1491,] 0.430813836 0.2910904300 6.556091e-02
## [1492,] 0.227598100 0.4443581954 2.891855e-01
## [1493,] 0.195398778 0.4422182874 3.336033e-01
## [1494,] 0.375000000 0.3750000000 1.250000e-01
## [1495,] 0.306334128 0.4241549461 1.957638e-01
## [1496,] 0.440355309 0.2596967205 5.105149e-02
## [1497,] 0.360146521 0.0776786613 5.584740e-03
## [1498,] 0.118425028 0.4060286664 4.640328e-01
## [1499,] 0.426168977 0.1482326877 1.718640e-02
## [1500,] 0.440355309 0.2596967205 5.105149e-02
## [1501,] 0.293645732 0.0435030714 2.148300e-03
## [1502,] 0.306334128 0.4241549461 1.957638e-01
## [1503,] 0.424154946 0.3063341278 7.374710e-02
## [1504,] 0.321175019 0.4163379880 1.798991e-01
## [1505,] 0.306334128 0.4241549461 1.957638e-01
## [1506,] 0.179450170 0.4386559699 3.574234e-01
## [1507,] 0.443086838 0.2436977611 4.467792e-02
## [1508,] 0.444358195 0.2275981001 3.885821e-02
## [1509,] 0.291090430 0.4308138364 2.125348e-01
## [1510,] 0.259696720 0.4403553087 2.488965e-01
## [1511,] 0.416337988 0.3211750193 8.258786e-02
## [1512,] 0.340371253 0.0654560102 4.195898e-03
## [1513,] 0.243697761 0.4430868383 2.685375e-01
## [1514,] 0.335537578 0.4074384881 1.649156e-01
## [1515,] 0.392899701 0.1042386963 9.218388e-03
## [1516,] 0.163702964 0.4333313752 3.823512e-01
## [1517,] 0.436239133 0.2755194522 5.800410e-02
## [1518,] 0.377630828 0.0906313987 7.250512e-03
## [1519,] 0.335537578 0.4074384881 1.649156e-01
## [1520,] 0.436239133 0.2755194522 5.800410e-02
## [1521,] 0.259696720 0.4403553087 2.488965e-01
## [1522,] 0.407438488 0.3355375785 9.210835e-02
## [1523,] 0.131453291 0.0066840657 1.132892e-04
## [1524,] 0.426168977 0.1482326877 1.718640e-02
## [1525,] 0.444358195 0.2275981001 3.885821e-02
## [1526,] 0.436239133 0.2755194522 5.800410e-02
## [1527,] 0.000000000 0.0000000000 1.000000e+00
## [1528,] 0.392899701 0.1042386963 9.218388e-03
## [1529,] 0.440355309 0.2596967205 5.105149e-02
## [1530,] 0.442218287 0.1953987782 2.877966e-02
## [1531,] 0.430813836 0.2910904300 6.556091e-02
## [1532,] 0.306334128 0.4241549461 1.957638e-01
## [1533,] 0.416337988 0.3211750193 8.258786e-02
## [1534,] 0.227598100 0.4443581954 2.891855e-01
## [1535,] 0.360146521 0.0776786613 5.584740e-03
## [1536,] 0.360146521 0.0776786613 5.584740e-03
## [1537,] 0.416337988 0.3211750193 8.258786e-02
## [1538,] 0.163702964 0.4333313752 3.823512e-01
## [1539,] 0.275519452 0.4362391326 2.302373e-01
## [1540,] 0.444358195 0.2275981001 3.885821e-02
## [1541,] 0.436239133 0.2755194522 5.800410e-02
## [1542,] 0.397531973 0.3493462791 1.023338e-01
## [1543,] 0.430813836 0.2910904300 6.556091e-02
## [1544,] 0.436239133 0.2755194522 5.800410e-02
## [1545,] 0.362525595 0.3866939680 1.374912e-01
## [1546,] 0.444358195 0.2275981001 3.885821e-02
## [1547,] 0.362525595 0.3866939680 1.374912e-01
## [1548,] 0.211473264 0.4440938538 3.108657e-01
## [1549,] 0.259696720 0.4403553087 2.488965e-01
## [1550,] 0.375000000 0.3750000000 1.250000e-01
## [1551,] 0.417093250 0.1331148669 1.416116e-02
## [1552,] 0.227598100 0.4443581954 2.891855e-01
## [1553,] 0.440355309 0.2596967205 5.105149e-02
## [1554,] 0.417093250 0.1331148669 1.416116e-02
## [1555,] 0.340371253 0.0654560102 4.195898e-03
## [1556,] 0.375000000 0.3750000000 1.250000e-01
## [1557,] 0.349346279 0.3975319727 1.507880e-01
## [1558,] 0.169380014 0.0116813803 2.685375e-04
## [1559,] 0.397531973 0.3493462791 1.023338e-01
## [1560,] 0.227598100 0.4443581954 2.891855e-01
## [1561,] 0.440355309 0.2596967205 5.105149e-02
## [1562,] 0.406028666 0.1184250277 1.151354e-02
## [1563,] 0.444358195 0.2275981001 3.885821e-02
## [1564,] 0.148232688 0.4261689772 4.084119e-01
## [1565,] 0.438655970 0.1794501695 2.447048e-02
## [1566,] 0.195398778 0.4422182874 3.336033e-01
## [1567,] 0.426168977 0.1482326877 1.718640e-02
## [1568,] 0.335537578 0.4074384881 1.649156e-01
## [1569,] 0.417093250 0.1331148669 1.416116e-02
## [1570,] 0.426168977 0.1482326877 1.718640e-02
## [1571,] 0.444358195 0.2275981001 3.885821e-02
## [1572,] 0.227598100 0.4443581954 2.891855e-01
## [1573,] 0.375000000 0.3750000000 1.250000e-01
## [1574,] 0.443086838 0.2436977611 4.467792e-02
## [1575,] 0.375000000 0.3750000000 1.250000e-01
## [1576,] 0.227598100 0.4443581954 2.891855e-01
## [1577,] 0.444358195 0.2275981001 3.885821e-02
## [1578,] 0.163702964 0.4333313752 3.823512e-01
## [1579,] 0.266544426 0.0339238361 1.439193e-03
## [1580,] 0.321175019 0.4163379880 1.798991e-01
## [1581,] 0.204487093 0.0179374643 5.244873e-04
## [1582,] 0.438655970 0.1794501695 2.447048e-02
## [1583,] 0.046838810 0.0007678494 4.195898e-06
## [1584,] 0.430813836 0.2910904300 6.556091e-02
## [1585,] 0.443086838 0.2436977611 4.467792e-02
## [1586,] 0.444093854 0.2114732637 3.356718e-02
## [1587,] 0.163702964 0.4333313752 3.823512e-01
## [1588,] 0.416337988 0.3211750193 8.258786e-02
## [1589,] 0.406028666 0.1184250277 1.151354e-02
## [1590,] 0.442218287 0.1953987782 2.877966e-02
## [1591,] 0.442218287 0.1953987782 2.877966e-02
## [1592,] 0.416337988 0.3211750193 8.258786e-02
## [1593,] 0.424154946 0.3063341278 7.374710e-02
## [1594,] 0.444358195 0.2275981001 3.885821e-02
## [1595,] 0.417093250 0.1331148669 1.416116e-02
## [1596,] 0.433331375 0.1637029640 2.061445e-02
## [1597,] 0.163702964 0.4333313752 3.823512e-01
## [1598,] 0.416337988 0.3211750193 8.258786e-02
## [1599,] 0.440355309 0.2596967205 5.105149e-02
## [1600,] 0.416337988 0.3211750193 8.258786e-02
## [1601,] 0.433331375 0.1637029640 2.061445e-02
## [1602,] 0.335537578 0.4074384881 1.649156e-01
## [1603,] 0.443086838 0.2436977611 4.467792e-02
## [1604,] 0.440355309 0.2596967205 5.105149e-02
## [1605,] 0.386693968 0.3625255950 1.132892e-01
## [1606,] 0.291090430 0.4308138364 2.125348e-01
## [1607,] 0.148232688 0.4261689772 4.084119e-01
## [1608,] 0.360146521 0.0776786613 5.584740e-03
## [1609,] 0.440355309 0.2596967205 5.105149e-02
## [1610,] 0.243697761 0.4430868383 2.685375e-01
## [1611,] 0.426168977 0.1482326877 1.718640e-02
## [1612,] 0.430813836 0.2910904300 6.556091e-02
## [1613,] 0.407438488 0.3355375785 9.210835e-02
## [1614,] 0.397531973 0.3493462791 1.023338e-01
## [1615,] 0.416337988 0.3211750193 8.258786e-02
## [1616,] 0.426168977 0.1482326877 1.718640e-02
## [1617,] 0.406028666 0.1184250277 1.151354e-02
## [1618,] 0.291090430 0.4308138364 2.125348e-01
## [1619,] 0.169380014 0.0116813803 2.685375e-04
## [1620,] 0.426168977 0.1482326877 1.718640e-02
## [1621,] 0.386693968 0.3625255950 1.132892e-01
## [1622,] 0.375000000 0.3750000000 1.250000e-01
## [1623,] 0.397531973 0.3493462791 1.023338e-01
## [1624,] 0.433331375 0.1637029640 2.061445e-02
## [1625,] 0.362525595 0.3866939680 1.374912e-01
## [1626,] 0.291090430 0.4308138364 2.125348e-01
## [1627,] 0.416337988 0.3211750193 8.258786e-02
## [1628,] 0.443086838 0.2436977611 4.467792e-02
## [1629,] 0.397531973 0.3493462791 1.023338e-01
## [1630,] 0.436239133 0.2755194522 5.800410e-02
## [1631,] 0.386693968 0.3625255950 1.132892e-01
## [1632,] 0.375000000 0.3750000000 1.250000e-01
## [1633,] 0.349346279 0.3975319727 1.507880e-01
## [1634,] 0.243697761 0.4430868383 2.685375e-01
## [1635,] 0.406028666 0.1184250277 1.151354e-02
## [1636,] 0.291090430 0.4308138364 2.125348e-01
## [1637,] 0.266544426 0.0339238361 1.439193e-03
## [1638,] 0.033923836 0.2665444262 6.980925e-01
## [1639,] 0.000000000 0.0000000000 0.000000e+00
## [1640,] 0.335537578 0.4074384881 1.649156e-01
## [1641,] 0.349346279 0.3975319727 1.507880e-01
## [1642,] 0.424154946 0.3063341278 7.374710e-02
## [1643,] 0.360146521 0.0776786613 5.584740e-03
## [1644,] 0.386693968 0.3625255950 1.132892e-01
## [1645,] 0.179450170 0.4386559699 3.574234e-01
## [1646,] 0.236850055 0.0253767916 9.063140e-04
## [1647,] 0.386693968 0.3625255950 1.132892e-01
## [1648,] 0.306334128 0.4241549461 1.957638e-01
## [1649,] 0.386693968 0.3625255950 1.132892e-01
## [1650,] 0.033923836 0.2665444262 6.980925e-01
## [1651,] 0.377630828 0.0906313987 7.250512e-03
## [1652,] 0.386693968 0.3625255950 1.132892e-01
## [1653,] 0.360146521 0.0776786613 5.584740e-03
## [1654,] 0.443086838 0.2436977611 4.467792e-02
## [1655,] 0.335537578 0.4074384881 1.649156e-01
## [1656,] 0.407438488 0.3355375785 9.210835e-02
## [1657,] 0.424154946 0.3063341278 7.374710e-02
## [1658,] 0.443086838 0.2436977611 4.467792e-02
## [1659,] 0.392899701 0.1042386963 9.218388e-03
## [1660,] 0.046838810 0.0007678494 4.195898e-06
## [1661,] 0.430813836 0.2910904300 6.556091e-02
## [1662,] 0.275519452 0.4362391326 2.302373e-01
## [1663,] 0.291090430 0.4308138364 2.125348e-01
## [1664,] 0.436239133 0.2755194522 5.800410e-02
## [1665,] 0.318229499 0.0540389715 3.058810e-03
## [1666,] 0.426168977 0.1482326877 1.718640e-02
## [1667,] 0.397531973 0.3493462791 1.023338e-01
## [1668,] 0.417093250 0.1331148669 1.416116e-02
## [1669,] 0.433331375 0.1637029640 2.061445e-02
## [1670,] 0.443086838 0.2436977611 4.467792e-02
## [1671,] 0.397531973 0.3493462791 1.023338e-01
## [1672,] 0.416337988 0.3211750193 8.258786e-02
## [1673,] 0.306334128 0.4241549461 1.957638e-01
## [1674,] 0.440355309 0.2596967205 5.105149e-02
## [1675,] 0.407438488 0.3355375785 9.210835e-02
## [1676,] 0.424154946 0.3063341278 7.374710e-02
## [1677,] 0.424154946 0.3063341278 7.374710e-02
## [1678,] 0.407438488 0.3355375785 9.210835e-02
## [1679,] 0.444093854 0.2114732637 3.356718e-02
## [1680,] 0.417093250 0.1331148669 1.416116e-02
## [1681,] 0.335537578 0.4074384881 1.649156e-01
## [1682,] 0.417093250 0.1331148669 1.416116e-02
## [1683,] 0.406028666 0.1184250277 1.151354e-02
## [1684,] 0.444358195 0.2275981001 3.885821e-02
## [1685,] 0.438655970 0.1794501695 2.447048e-02
## [1686,] 0.442218287 0.1953987782 2.877966e-02
## [1687,] 0.443086838 0.2436977611 4.467792e-02
## [1688,] 0.275519452 0.4362391326 2.302373e-01
## [1689,] 0.375000000 0.3750000000 1.250000e-01
## [1690,] 0.406028666 0.1184250277 1.151354e-02
## [1691,] 0.386693968 0.3625255950 1.132892e-01
## [1692,] 0.386693968 0.3625255950 1.132892e-01
## [1693,] 0.406028666 0.1184250277 1.151354e-02
## [1694,] 0.377630828 0.0906313987 7.250512e-03
## [1695,] 0.417093250 0.1331148669 1.416116e-02
## [1696,] 0.275519452 0.4362391326 2.302373e-01
## [1697,] 0.407438488 0.3355375785 9.210835e-02
## [1698,] 0.375000000 0.3750000000 1.250000e-01
## [1699,] 0.442218287 0.1953987782 2.877966e-02
## [1700,] 0.321175019 0.4163379880 1.798991e-01
## [1701,] 0.275519452 0.4362391326 2.302373e-01
## [1702,] 0.275519452 0.4362391326 2.302373e-01
## [1703,] 0.386693968 0.3625255950 1.132892e-01
## [1704,] 0.397531973 0.3493462791 1.023338e-01
## [1705,] 0.335537578 0.4074384881 1.649156e-01
## [1706,] 0.443086838 0.2436977611 4.467792e-02
## [1707,] 0.433331375 0.1637029640 2.061445e-02
## [1708,] 0.443086838 0.2436977611 4.467792e-02
## [1709,] 0.169380014 0.0116813803 2.685375e-04
## [1710,] 0.386693968 0.3625255950 1.132892e-01
## [1711,] 0.443086838 0.2436977611 4.467792e-02
## [1712,] 0.416337988 0.3211750193 8.258786e-02
## [1713,] 0.377630828 0.0906313987 7.250512e-03
## [1714,] 0.407438488 0.3355375785 9.210835e-02
## [1715,] 0.406028666 0.1184250277 1.151354e-02
## [1716,] 0.321175019 0.4163379880 1.798991e-01
## [1717,] 0.406028666 0.1184250277 1.151354e-02
## [1718,] 0.444358195 0.2275981001 3.885821e-02
## [1719,] 0.349346279 0.3975319727 1.507880e-01
## [1720,] 0.443086838 0.2436977611 4.467792e-02
## [1721,] 0.118425028 0.4060286664 4.640328e-01
## [1722,] 0.443086838 0.2436977611 4.467792e-02
## [1723,] 0.335537578 0.4074384881 1.649156e-01
## [1724,] 0.406028666 0.1184250277 1.151354e-02
## [1725,] 0.416337988 0.3211750193 8.258786e-02
## [1726,] 0.442218287 0.1953987782 2.877966e-02
## [1727,] 0.375000000 0.3750000000 1.250000e-01
## [1728,] 0.321175019 0.4163379880 1.798991e-01
## [1729,] 0.118425028 0.4060286664 4.640328e-01
## [1730,] 0.440355309 0.2596967205 5.105149e-02
## [1731,] 0.306334128 0.4241549461 1.957638e-01
## [1732,] 0.236850055 0.0253767916 9.063140e-04
## [1733,] 0.179450170 0.4386559699 3.574234e-01
## [1734,] 0.163702964 0.4333313752 3.823512e-01
## [1735,] 0.293645732 0.0435030714 2.148300e-03
## [1736,] 0.416337988 0.3211750193 8.258786e-02
## [1737,] 0.204487093 0.0179374643 5.244873e-04
## [1738,] 0.392899701 0.1042386963 9.218388e-03
## [1739,] 0.430813836 0.2910904300 6.556091e-02
## [1740,] 0.386693968 0.3625255950 1.132892e-01
## [1741,] 0.291090430 0.4308138364 2.125348e-01
## [1742,] 0.386693968 0.3625255950 1.132892e-01
## [1743,] 0.163702964 0.4333313752 3.823512e-01
## [1744,] 0.259696720 0.4403553087 2.488965e-01
## [1745,] 0.077678661 0.3601465208 5.565901e-01
## [1746,] 0.392899701 0.1042386963 9.218388e-03
## [1747,] 0.444093854 0.2114732637 3.356718e-02
## [1748,] 0.424154946 0.3063341278 7.374710e-02
## [1749,] 0.392899701 0.1042386963 9.218388e-03
## [1750,] 0.375000000 0.3750000000 1.250000e-01
## [1751,] 0.293645732 0.0435030714 2.148300e-03
## [1752,] 0.377630828 0.0906313987 7.250512e-03
## [1753,] 0.443086838 0.2436977611 4.467792e-02
## [1754,] 0.424154946 0.3063341278 7.374710e-02
## [1755,] 0.133114867 0.4170932496 4.356307e-01
## [1756,] 0.306334128 0.4241549461 1.957638e-01
## [1757,] 0.275519452 0.4362391326 2.302373e-01
## [1758,] 0.442218287 0.1953987782 2.877966e-02
## [1759,] 0.407438488 0.3355375785 9.210835e-02
## [1760,] 0.442218287 0.1953987782 2.877966e-02
## [1761,] 0.243697761 0.4430868383 2.685375e-01
## [1762,] 0.349346279 0.3975319727 1.507880e-01
## [1763,] 0.436239133 0.2755194522 5.800410e-02
## [1764,] 0.407438488 0.3355375785 9.210835e-02
## [1765,] 0.430813836 0.2910904300 6.556091e-02
## [1766,] 0.397531973 0.3493462791 1.023338e-01
## [1767,] 0.424154946 0.3063341278 7.374710e-02
## [1768,] 0.438655970 0.1794501695 2.447048e-02
## [1769,] 0.360146521 0.0776786613 5.584740e-03
## [1770,] 0.090631399 0.0030210466 3.356718e-05
## [1771,] 0.406028666 0.1184250277 1.151354e-02
## [1772,] 0.438655970 0.1794501695 2.447048e-02
## [1773,] 0.392899701 0.1042386963 9.218388e-03
## [1774,] 0.340371253 0.0654560102 4.195898e-03
## [1775,] 0.436239133 0.2755194522 5.800410e-02
## [1776,] 0.148232688 0.4261689772 4.084119e-01
## [1777,] 0.442218287 0.1953987782 2.877966e-02
## [1778,] 0.377630828 0.0906313987 7.250512e-03
## [1779,] 0.293645732 0.0435030714 2.148300e-03
## [1780,] 0.424154946 0.3063341278 7.374710e-02
## [1781,] 0.386693968 0.3625255950 1.132892e-01
## [1782,] 0.321175019 0.4163379880 1.798991e-01
## [1783,] 0.436239133 0.2755194522 5.800410e-02
## [1784,] 0.266544426 0.0339238361 1.439193e-03
## [1785,] 0.335537578 0.4074384881 1.649156e-01
## [1786,] 0.444093854 0.2114732637 3.356718e-02
## [1787,] 0.360146521 0.0776786613 5.584740e-03
## [1788,] 0.259696720 0.4403553087 2.488965e-01
## [1789,] 0.362525595 0.3866939680 1.374912e-01
## [1790,] 0.204487093 0.0179374643 5.244873e-04
## [1791,] 0.195398778 0.4422182874 3.336033e-01
## [1792,] 0.065456010 0.3403712531 5.899768e-01
## [1793,] 0.227598100 0.4443581954 2.891855e-01
## [1794,] 0.266544426 0.0339238361 1.439193e-03
## [1795,] 0.386693968 0.3625255950 1.132892e-01
## [1796,] 0.335537578 0.4074384881 1.649156e-01
## [1797,] 0.424154946 0.3063341278 7.374710e-02
## [1798,] 0.430813836 0.2910904300 6.556091e-02
## [1799,] 0.349346279 0.3975319727 1.507880e-01
## [1800,] 0.430813836 0.2910904300 6.556091e-02
## [1801,] 0.340371253 0.0654560102 4.195898e-03
## [1802,] 0.306334128 0.4241549461 1.957638e-01
## [1803,] 0.438655970 0.1794501695 2.447048e-02
## [1804,] 0.054038972 0.3182294988 6.246727e-01
## [1805,] 0.204487093 0.0179374643 5.244873e-04
## [1806,] 0.436239133 0.2755194522 5.800410e-02
## [1807,] 0.318229499 0.0540389715 3.058810e-03
## [1808,] 0.360146521 0.0776786613 5.584740e-03
## [1809,] 0.440355309 0.2596967205 5.105149e-02
## [1810,] 0.169380014 0.0116813803 2.685375e-04
## [1811,] 0.444358195 0.2275981001 3.885821e-02
## [1812,] 0.375000000 0.3750000000 1.250000e-01
## [1813,] 0.436239133 0.2755194522 5.800410e-02
## [1814,] 0.291090430 0.4308138364 2.125348e-01
## [1815,] 0.397531973 0.3493462791 1.023338e-01
## [1816,] 0.377630828 0.0906313987 7.250512e-03
## [1817,] 0.275519452 0.4362391326 2.302373e-01
## [1818,] 0.430813836 0.2910904300 6.556091e-02
## [1819,] 0.433331375 0.1637029640 2.061445e-02
## [1820,] 0.243697761 0.4430868383 2.685375e-01
## [1821,] 0.077678661 0.3601465208 5.565901e-01
## [1822,] 0.090631399 0.3776308281 5.244873e-01
## [1823,] 0.335537578 0.4074384881 1.649156e-01
## [1824,] 0.118425028 0.4060286664 4.640328e-01
## [1825,] 0.377630828 0.0906313987 7.250512e-03
## [1826,] 0.430813836 0.2910904300 6.556091e-02
## [1827,] 0.306334128 0.4241549461 1.957638e-01
## [1828,] 0.442218287 0.1953987782 2.877966e-02
## [1829,] 0.407438488 0.3355375785 9.210835e-02
## [1830,] 0.321175019 0.4163379880 1.798991e-01
## [1831,] 0.392899701 0.1042386963 9.218388e-03
## [1832,] 0.000000000 0.0000000000 0.000000e+00
## [1833,] 0.375000000 0.3750000000 1.250000e-01
## [1834,] 0.443086838 0.2436977611 4.467792e-02
## [1835,] 0.433331375 0.1637029640 2.061445e-02
## [1836,] 0.407438488 0.3355375785 9.210835e-02
## [1837,] 0.443086838 0.2436977611 4.467792e-02
## [1838,] 0.444358195 0.2275981001 3.885821e-02
## [1839,] 0.436239133 0.2755194522 5.800410e-02
## [1840,] 0.442218287 0.1953987782 2.877966e-02
## [1841,] 0.243697761 0.4430868383 2.685375e-01
## [1842,] 0.443086838 0.2436977611 4.467792e-02
## [1843,] 0.318229499 0.0540389715 3.058810e-03
## [1844,] 0.392899701 0.1042386963 9.218388e-03
## [1845,] 0.424154946 0.3063341278 7.374710e-02
## [1846,] 0.444093854 0.2114732637 3.356718e-02
## [1847,] 0.426168977 0.1482326877 1.718640e-02
## [1848,] 0.440355309 0.2596967205 5.105149e-02
## [1849,] 0.090631399 0.0030210466 3.356718e-05
## [1850,] 0.444093854 0.2114732637 3.356718e-02
## [1851,] 0.430813836 0.2910904300 6.556091e-02
## [1852,] 0.362525595 0.3866939680 1.374912e-01
## [1853,] 0.291090430 0.4308138364 2.125348e-01
## [1854,] 0.236850055 0.0253767916 9.063140e-04
## [1855,] 0.440355309 0.2596967205 5.105149e-02
## [1856,] 0.442218287 0.1953987782 2.877966e-02
## [1857,] 0.436239133 0.2755194522 5.800410e-02
## [1858,] 0.266544426 0.0339238361 1.439193e-03
## [1859,] 0.416337988 0.3211750193 8.258786e-02
## [1860,] 0.443086838 0.2436977611 4.467792e-02
## [1861,] 0.430813836 0.2910904300 6.556091e-02
## [1862,] 0.362525595 0.3866939680 1.374912e-01
## [1863,] 0.436239133 0.2755194522 5.800410e-02
## [1864,] 0.046838810 0.0007678494 4.195898e-06
## [1865,] 0.424154946 0.3063341278 7.374710e-02
## [1866,] 0.293645732 0.0435030714 2.148300e-03
## [1867,] 0.306334128 0.4241549461 1.957638e-01
## [1868,] 0.406028666 0.1184250277 1.151354e-02
## [1869,] 0.375000000 0.3750000000 1.250000e-01
## [1870,] 0.433331375 0.1637029640 2.061445e-02
## [1871,] 0.426168977 0.1482326877 1.718640e-02
## [1872,] 0.204487093 0.0179374643 5.244873e-04
## [1873,] 0.211473264 0.4440938538 3.108657e-01
## [1874,] 0.397531973 0.3493462791 1.023338e-01
## [1875,] 0.386693968 0.3625255950 1.132892e-01
## [1876,] 0.433331375 0.1637029640 2.061445e-02
## [1877,] 0.291090430 0.4308138364 2.125348e-01
## [1878,] 0.433331375 0.1637029640 2.061445e-02
## [1879,] 0.442218287 0.1953987782 2.877966e-02
## [1880,] 0.318229499 0.0540389715 3.058810e-03
## [1881,] 0.148232688 0.4261689772 4.084119e-01
## [1882,] 0.293645732 0.0435030714 2.148300e-03
## [1883,] 0.440355309 0.2596967205 5.105149e-02
## [1884,] 0.169380014 0.0116813803 2.685375e-04
## [1885,] 0.407438488 0.3355375785 9.210835e-02
## [1886,] 0.204487093 0.0179374643 5.244873e-04
## [1887,] 0.424154946 0.3063341278 7.374710e-02
## [1888,] 0.090631399 0.0030210466 3.356718e-05
## [1889,] 0.430813836 0.2910904300 6.556091e-02
## [1890,] 0.407438488 0.3355375785 9.210835e-02
## [1891,] 0.417093250 0.1331148669 1.416116e-02
## [1892,] 0.179450170 0.4386559699 3.574234e-01
## [1893,] 0.444093854 0.2114732637 3.356718e-02
## [1894,] 0.407438488 0.3355375785 9.210835e-02
## [1895,] 0.163702964 0.4333313752 3.823512e-01
## [1896,] 0.243697761 0.4430868383 2.685375e-01
## [1897,] 0.204487093 0.0179374643 5.244873e-04
## [1898,] 0.362525595 0.3866939680 1.374912e-01
## [1899,] 0.433331375 0.1637029640 2.061445e-02
## [1900,] 0.444093854 0.2114732637 3.356718e-02
## [1901,] 0.438655970 0.1794501695 2.447048e-02
## [1902,] 0.406028666 0.1184250277 1.151354e-02
## [1903,] 0.440355309 0.2596967205 5.105149e-02
## [1904,] 0.293645732 0.0435030714 2.148300e-03
## [1905,] 0.293645732 0.0435030714 2.148300e-03
## [1906,] 0.266544426 0.0339238361 1.439193e-03
## [1907,] 0.243697761 0.4430868383 2.685375e-01
## [1908,] 0.259696720 0.4403553087 2.488965e-01
## [1909,] 0.377630828 0.0906313987 7.250512e-03
## [1910,] 0.424154946 0.3063341278 7.374710e-02
## [1911,] 0.360146521 0.0776786613 5.584740e-03
## [1912,] 0.349346279 0.3975319727 1.507880e-01
## [1913,] 0.442218287 0.1953987782 2.877966e-02
## [1914,] 0.104238696 0.3928997013 4.936432e-01
## [1915,] 0.426168977 0.1482326877 1.718640e-02
## [1916,] 0.362525595 0.3866939680 1.374912e-01
## [1917,] 0.444093854 0.2114732637 3.356718e-02
## [1918,] 0.291090430 0.4308138364 2.125348e-01
## [1919,] 0.444358195 0.2275981001 3.885821e-02
## [1920,] 0.306334128 0.4241549461 1.957638e-01
## [1921,] 0.375000000 0.3750000000 1.250000e-01
## [1922,] 0.444358195 0.2275981001 3.885821e-02
## [1923,] 0.406028666 0.1184250277 1.151354e-02
## [1924,] 0.397531973 0.3493462791 1.023338e-01
## [1925,] 0.443086838 0.2436977611 4.467792e-02
## [1926,] 0.349346279 0.3975319727 1.507880e-01
## [1927,] 0.340371253 0.0654560102 4.195898e-03
## [1928,] 0.291090430 0.4308138364 2.125348e-01
## [1929,] 0.424154946 0.3063341278 7.374710e-02
## [1930,] 0.377630828 0.0906313987 7.250512e-03
## [1931,] 0.443086838 0.2436977611 4.467792e-02
## [1932,] 0.375000000 0.3750000000 1.250000e-01
## [1933,] 0.430813836 0.2910904300 6.556091e-02
## [1934,] 0.424154946 0.3063341278 7.374710e-02
## [1935,] 0.406028666 0.1184250277 1.151354e-02
## [1936,] 0.426168977 0.1482326877 1.718640e-02
## [1937,] 0.438655970 0.1794501695 2.447048e-02
## [1938,] 0.349346279 0.3975319727 1.507880e-01
## [1939,] 0.211473264 0.4440938538 3.108657e-01
## [1940,] 0.438655970 0.1794501695 2.447048e-02
## [1941,] 0.440355309 0.2596967205 5.105149e-02
## [1942,] 0.275519452 0.4362391326 2.302373e-01
## [1943,] 0.424154946 0.3063341278 7.374710e-02
## [1944,] 0.416337988 0.3211750193 8.258786e-02
## [1945,] 0.266544426 0.0339238361 1.439193e-03
## [1946,] 0.335537578 0.4074384881 1.649156e-01
## [1947,] 0.377630828 0.0906313987 7.250512e-03
## [1948,] 0.360146521 0.0776786613 5.584740e-03
## [1949,] 0.204487093 0.0179374643 5.244873e-04
## [1950,] 0.386693968 0.3625255950 1.132892e-01
## [1951,] 0.424154946 0.3063341278 7.374710e-02
## [1952,] 0.349346279 0.3975319727 1.507880e-01
## [1953,] 0.438655970 0.1794501695 2.447048e-02
## [1954,] 0.204487093 0.0179374643 5.244873e-04
## [1955,] 0.349346279 0.3975319727 1.507880e-01
## [1956,] 0.397531973 0.3493462791 1.023338e-01
## [1957,] 0.426168977 0.1482326877 1.718640e-02
## [1958,] 0.426168977 0.1482326877 1.718640e-02
## [1959,] 0.430813836 0.2910904300 6.556091e-02
## [1960,] 0.430813836 0.2910904300 6.556091e-02
## [1961,] 0.227598100 0.4443581954 2.891855e-01
## [1962,] 0.321175019 0.4163379880 1.798991e-01
## [1963,] 0.090631399 0.0030210466 3.356718e-05
## [1964,] 0.443086838 0.2436977611 4.467792e-02
## [1965,] 0.386693968 0.3625255950 1.132892e-01
## [1966,] 0.430813836 0.2910904300 6.556091e-02
## [1967,] 0.275519452 0.4362391326 2.302373e-01
## [1968,] 0.291090430 0.4308138364 2.125348e-01
## [1969,] 0.444093854 0.2114732637 3.356718e-02
## [1970,] 0.335537578 0.4074384881 1.649156e-01
## [1971,] 0.443086838 0.2436977611 4.467792e-02
## [1972,] 0.360146521 0.0776786613 5.584740e-03
## [1973,] 0.444358195 0.2275981001 3.885821e-02
## [1974,] 0.362525595 0.3866939680 1.374912e-01
## [1975,] 0.362525595 0.3866939680 1.374912e-01
## [1976,] 0.259696720 0.4403553087 2.488965e-01
## [1977,] 0.377630828 0.0906313987 7.250512e-03
## [1978,] 0.275519452 0.4362391326 2.302373e-01
## [1979,] 0.104238696 0.3928997013 4.936432e-01
## [1980,] 0.349346279 0.3975319727 1.507880e-01
## [1981,] 0.416337988 0.3211750193 8.258786e-02
## [1982,] 0.306334128 0.4241549461 1.957638e-01
## [1983,] 0.204487093 0.0179374643 5.244873e-04
## [1984,] 0.025376792 0.2368500554 7.368668e-01
## [1985,] 0.442218287 0.1953987782 2.877966e-02
## [1986,] 0.291090430 0.4308138364 2.125348e-01
## [1987,] 0.266544426 0.0339238361 1.439193e-03
## [1988,] 0.118425028 0.4060286664 4.640328e-01
## [1989,] 0.163702964 0.4333313752 3.823512e-01
## [1990,] 0.424154946 0.3063341278 7.374710e-02
## [1991,] 0.406028666 0.1184250277 1.151354e-02
## [1992,] 0.430813836 0.2910904300 6.556091e-02
## [1993,] 0.442218287 0.1953987782 2.877966e-02
## [1994,] 0.293645732 0.0435030714 2.148300e-03
## [1995,] 0.444358195 0.2275981001 3.885821e-02
## [1996,] 0.416337988 0.3211750193 8.258786e-02
## [1997,] 0.443086838 0.2436977611 4.467792e-02
## [1998,] 0.349346279 0.3975319727 1.507880e-01
## [1999,] 0.430813836 0.2910904300 6.556091e-02
## [2000,] 0.335537578 0.4074384881 1.649156e-01
## [2001,] 0.362525595 0.3866939680 1.374912e-01
## [2002,] 0.306334128 0.4241549461 1.957638e-01
## [2003,] 0.340371253 0.0654560102 4.195898e-03
## [2004,] 0.340371253 0.0654560102 4.195898e-03
## [2005,] 0.293645732 0.0435030714 2.148300e-03
## [2006,] 0.416337988 0.3211750193 8.258786e-02
## [2007,] 0.033923836 0.2665444262 6.980925e-01
## [2008,] 0.392899701 0.1042386963 9.218388e-03
## [2009,] 0.443086838 0.2436977611 4.467792e-02
## [2010,] 0.444093854 0.2114732637 3.356718e-02
## [2011,] 0.436239133 0.2755194522 5.800410e-02
## [2012,] 0.362525595 0.3866939680 1.374912e-01
## [2013,] 0.349346279 0.3975319727 1.507880e-01
## [2014,] 0.443086838 0.2436977611 4.467792e-02
## [2015,] 0.266544426 0.0339238361 1.439193e-03
## [2016,] 0.397531973 0.3493462791 1.023338e-01
## [2017,] 0.104238696 0.3928997013 4.936432e-01
## [2018,] 0.424154946 0.3063341278 7.374710e-02
## [2019,] 0.417093250 0.1331148669 1.416116e-02
## [2020,] 0.360146521 0.0776786613 5.584740e-03
## [2021,] 0.318229499 0.0540389715 3.058810e-03
## [2022,] 0.443086838 0.2436977611 4.467792e-02
## [2023,] 0.438655970 0.1794501695 2.447048e-02
## [2024,] 0.386693968 0.3625255950 1.132892e-01
## [2025,] 0.321175019 0.4163379880 1.798991e-01
## [2026,] 0.444093854 0.2114732637 3.356718e-02
## [2027,] 0.065456010 0.3403712531 5.899768e-01
## [2028,] 0.236850055 0.0253767916 9.063140e-04
## [2029,] 0.169380014 0.0116813803 2.685375e-04
## [2030,] 0.360146521 0.0776786613 5.584740e-03
## [2031,] 0.444093854 0.2114732637 3.356718e-02
## [2032,] 0.054038972 0.3182294988 6.246727e-01
## [2033,] 0.406028666 0.1184250277 1.151354e-02
## [2034,] 0.406028666 0.1184250277 1.151354e-02
## [2035,] 0.417093250 0.1331148669 1.416116e-02
## [2036,] 0.438655970 0.1794501695 2.447048e-02
## [2037,] 0.407438488 0.3355375785 9.210835e-02
## [2038,] 0.227598100 0.4443581954 2.891855e-01
## [2039,] 0.377630828 0.0906313987 7.250512e-03
## [2040,] 0.306334128 0.4241549461 1.957638e-01
## [2041,] 0.392899701 0.1042386963 9.218388e-03
## [2042,] 0.426168977 0.1482326877 1.718640e-02
## [2043,] 0.397531973 0.3493462791 1.023338e-01
## [2044,] 0.360146521 0.0776786613 5.584740e-03
## [2045,] 0.243697761 0.4430868383 2.685375e-01
## [2046,] 0.440355309 0.2596967205 5.105149e-02
## [2047,] 0.275519452 0.4362391326 2.302373e-01
## [2048,] 0.335537578 0.4074384881 1.649156e-01
## [2049,] 0.321175019 0.4163379880 1.798991e-01
## [2050,] 0.442218287 0.1953987782 2.877966e-02
## [2051,] 0.433331375 0.1637029640 2.061445e-02
## [2052,] 0.443086838 0.2436977611 4.467792e-02
## [2053,] 0.306334128 0.4241549461 1.957638e-01
## [2054,] 0.442218287 0.1953987782 2.877966e-02
## [2055,] 0.444358195 0.2275981001 3.885821e-02
## [2056,] 0.397531973 0.3493462791 1.023338e-01
## [2057,] 0.349346279 0.3975319727 1.507880e-01
## [2058,] 0.397531973 0.3493462791 1.023338e-01
## [2059,] 0.340371253 0.0654560102 4.195898e-03
## [2060,] 0.133114867 0.4170932496 4.356307e-01
## [2061,] 0.436239133 0.2755194522 5.800410e-02
## [2062,] 0.243697761 0.4430868383 2.685375e-01
## [2063,] 0.375000000 0.3750000000 1.250000e-01
## [2064,] 0.424154946 0.3063341278 7.374710e-02
## [2065,] 0.386693968 0.3625255950 1.132892e-01
## [2066,] 0.436239133 0.2755194522 5.800410e-02
## [2067,] 0.377630828 0.0906313987 7.250512e-03
## [2068,] 0.392899701 0.1042386963 9.218388e-03
## [2069,] 0.360146521 0.0776786613 5.584740e-03
## [2070,] 0.442218287 0.1953987782 2.877966e-02
## [2071,] 0.275519452 0.4362391326 2.302373e-01
## [2072,] 0.424154946 0.3063341278 7.374710e-02
## [2073,] 0.266544426 0.0339238361 1.439193e-03
## [2074,] 0.392899701 0.1042386963 9.218388e-03
## [2075,] 0.349346279 0.3975319727 1.507880e-01
## [2076,] 0.266544426 0.0339238361 1.439193e-03
## [2077,] 0.362525595 0.3866939680 1.374912e-01
## [2078,] 0.377630828 0.0906313987 7.250512e-03
## [2079,] 0.443086838 0.2436977611 4.467792e-02
## [2080,] 0.426168977 0.1482326877 1.718640e-02
## [2081,] 0.436239133 0.2755194522 5.800410e-02
## [2082,] 0.377630828 0.0906313987 7.250512e-03
## [2083,] 0.293645732 0.0435030714 2.148300e-03
## [2084,] 0.360146521 0.0776786613 5.584740e-03
## [2085,] 0.306334128 0.4241549461 1.957638e-01
## [2086,] 0.349346279 0.3975319727 1.507880e-01
## [2087,] 0.375000000 0.3750000000 1.250000e-01
## [2088,] 0.321175019 0.4163379880 1.798991e-01
## [2089,] 0.443086838 0.2436977611 4.467792e-02
## [2090,] 0.335537578 0.4074384881 1.649156e-01
## [2091,] 0.275519452 0.4362391326 2.302373e-01
## [2092,] 0.377630828 0.0906313987 7.250512e-03
## [2093,] 0.349346279 0.3975319727 1.507880e-01
## [2094,] 0.406028666 0.1184250277 1.151354e-02
## [2095,] 0.362525595 0.3866939680 1.374912e-01
## [2096,] 0.293645732 0.0435030714 2.148300e-03
## [2097,] 0.392899701 0.1042386963 9.218388e-03
## [2098,] 0.392899701 0.1042386963 9.218388e-03
## [2099,] 0.424154946 0.3063341278 7.374710e-02
## [2100,] 0.377630828 0.0906313987 7.250512e-03
## [2101,] 0.318229499 0.0540389715 3.058810e-03
## [2102,] 0.291090430 0.4308138364 2.125348e-01
## attr(,"degree")
## [1] 3
## attr(,"knots")
## numeric(0)
## attr(,"Boundary.knots")
## [1] 18 80
## attr(,"intercept")
## [1] FALSE
## attr(,"class")
## [1] "bs"     "basis"  "matrix"
```

```r
lm1<- lm(wage ~ bsBasis, data=training)
plot(training$age, training$wage, pch=19, cex=0.5)
points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)
```

![](lecture_2_files/figure-html/unnamed-chunk-8-1.png) 

```r
predict(bsBasis, age=testing$age)
```

```
##                   1            2            3
##    [1,] 0.236850055 0.0253767916 9.063140e-04
##    [2,] 0.416337988 0.3211750193 8.258786e-02
##    [3,] 0.430813836 0.2910904300 6.556091e-02
##    [4,] 0.362525595 0.3866939680 1.374912e-01
##    [5,] 0.306334128 0.4241549461 1.957638e-01
##    [6,] 0.424154946 0.3063341278 7.374710e-02
##    [7,] 0.377630828 0.0906313987 7.250512e-03
##    [8,] 0.444358195 0.2275981001 3.885821e-02
##    [9,] 0.442218287 0.1953987782 2.877966e-02
##   [10,] 0.362525595 0.3866939680 1.374912e-01
##   [11,] 0.275519452 0.4362391326 2.302373e-01
##   [12,] 0.444093854 0.2114732637 3.356718e-02
##   [13,] 0.443086838 0.2436977611 4.467792e-02
##   [14,] 0.375000000 0.3750000000 1.250000e-01
##   [15,] 0.430813836 0.2910904300 6.556091e-02
##   [16,] 0.426168977 0.1482326877 1.718640e-02
##   [17,] 0.000000000 0.0000000000 0.000000e+00
##   [18,] 0.291090430 0.4308138364 2.125348e-01
##   [19,] 0.349346279 0.3975319727 1.507880e-01
##   [20,] 0.417093250 0.1331148669 1.416116e-02
##   [21,] 0.426168977 0.1482326877 1.718640e-02
##   [22,] 0.438655970 0.1794501695 2.447048e-02
##   [23,] 0.275519452 0.4362391326 2.302373e-01
##   [24,] 0.266544426 0.0339238361 1.439193e-03
##   [25,] 0.406028666 0.1184250277 1.151354e-02
##   [26,] 0.318229499 0.0540389715 3.058810e-03
##   [27,] 0.340371253 0.0654560102 4.195898e-03
##   [28,] 0.318229499 0.0540389715 3.058810e-03
##   [29,] 0.430813836 0.2910904300 6.556091e-02
##   [30,] 0.362525595 0.3866939680 1.374912e-01
##   [31,] 0.444358195 0.2275981001 3.885821e-02
##   [32,] 0.259696720 0.4403553087 2.488965e-01
##   [33,] 0.266544426 0.0339238361 1.439193e-03
##   [34,] 0.430813836 0.2910904300 6.556091e-02
##   [35,] 0.204487093 0.0179374643 5.244873e-04
##   [36,] 0.377630828 0.0906313987 7.250512e-03
##   [37,] 0.195398778 0.4422182874 3.336033e-01
##   [38,] 0.426168977 0.1482326877 1.718640e-02
##   [39,] 0.077678661 0.3601465208 5.565901e-01
##   [40,] 0.386693968 0.3625255950 1.132892e-01
##   [41,] 0.375000000 0.3750000000 1.250000e-01
##   [42,] 0.436239133 0.2755194522 5.800410e-02
##   [43,] 0.442218287 0.1953987782 2.877966e-02
##   [44,] 0.131453291 0.0066840657 1.132892e-04
##   [45,] 0.243697761 0.4430868383 2.685375e-01
##   [46,] 0.266544426 0.0339238361 1.439193e-03
##   [47,] 0.443086838 0.2436977611 4.467792e-02
##   [48,] 0.424154946 0.3063341278 7.374710e-02
##   [49,] 0.424154946 0.3063341278 7.374710e-02
##   [50,] 0.195398778 0.4422182874 3.336033e-01
##   [51,] 0.291090430 0.4308138364 2.125348e-01
##   [52,] 0.436239133 0.2755194522 5.800410e-02
##   [53,] 0.266544426 0.0339238361 1.439193e-03
##   [54,] 0.321175019 0.4163379880 1.798991e-01
##   [55,] 0.397531973 0.3493462791 1.023338e-01
##   [56,] 0.407438488 0.3355375785 9.210835e-02
##   [57,] 0.426168977 0.1482326877 1.718640e-02
##   [58,] 0.169380014 0.0116813803 2.685375e-04
##   [59,] 0.416337988 0.3211750193 8.258786e-02
##   [60,] 0.179450170 0.4386559699 3.574234e-01
##   [61,] 0.306334128 0.4241549461 1.957638e-01
##   [62,] 0.426168977 0.1482326877 1.718640e-02
##   [63,] 0.362525595 0.3866939680 1.374912e-01
##   [64,] 0.407438488 0.3355375785 9.210835e-02
##   [65,] 0.440355309 0.2596967205 5.105149e-02
##   [66,] 0.444093854 0.2114732637 3.356718e-02
##   [67,] 0.433331375 0.1637029640 2.061445e-02
##   [68,] 0.118425028 0.4060286664 4.640328e-01
##   [69,] 0.442218287 0.1953987782 2.877966e-02
##   [70,] 0.444358195 0.2275981001 3.885821e-02
##   [71,] 0.436239133 0.2755194522 5.800410e-02
##   [72,] 0.349346279 0.3975319727 1.507880e-01
##   [73,] 0.444093854 0.2114732637 3.356718e-02
##   [74,] 0.375000000 0.3750000000 1.250000e-01
##   [75,] 0.436239133 0.2755194522 5.800410e-02
##   [76,] 0.430813836 0.2910904300 6.556091e-02
##   [77,] 0.227598100 0.4443581954 2.891855e-01
##   [78,] 0.259696720 0.4403553087 2.488965e-01
##   [79,] 0.266544426 0.0339238361 1.439193e-03
##   [80,] 0.375000000 0.3750000000 1.250000e-01
##   [81,] 0.444093854 0.2114732637 3.356718e-02
##   [82,] 0.195398778 0.4422182874 3.336033e-01
##   [83,] 0.335537578 0.4074384881 1.649156e-01
##   [84,] 0.211473264 0.4440938538 3.108657e-01
##   [85,] 0.407438488 0.3355375785 9.210835e-02
##   [86,] 0.131453291 0.0066840657 1.132892e-04
##   [87,] 0.195398778 0.4422182874 3.336033e-01
##   [88,] 0.406028666 0.1184250277 1.151354e-02
##   [89,] 0.243697761 0.4430868383 2.685375e-01
##   [90,] 0.406028666 0.1184250277 1.151354e-02
##   [91,] 0.169380014 0.0116813803 2.685375e-04
##   [92,] 0.349346279 0.3975319727 1.507880e-01
##   [93,] 0.424154946 0.3063341278 7.374710e-02
##   [94,] 0.211473264 0.4440938538 3.108657e-01
##   [95,] 0.443086838 0.2436977611 4.467792e-02
##   [96,] 0.433331375 0.1637029640 2.061445e-02
##   [97,] 0.433331375 0.1637029640 2.061445e-02
##   [98,] 0.211473264 0.4440938538 3.108657e-01
##   [99,] 0.444093854 0.2114732637 3.356718e-02
##  [100,] 0.321175019 0.4163379880 1.798991e-01
##  [101,] 0.259696720 0.4403553087 2.488965e-01
##  [102,] 0.148232688 0.4261689772 4.084119e-01
##  [103,] 0.433331375 0.1637029640 2.061445e-02
##  [104,] 0.306334128 0.4241549461 1.957638e-01
##  [105,] 0.416337988 0.3211750193 8.258786e-02
##  [106,] 0.243697761 0.4430868383 2.685375e-01
##  [107,] 0.386693968 0.3625255950 1.132892e-01
##  [108,] 0.407438488 0.3355375785 9.210835e-02
##  [109,] 0.407438488 0.3355375785 9.210835e-02
##  [110,] 0.291090430 0.4308138364 2.125348e-01
##  [111,] 0.349346279 0.3975319727 1.507880e-01
##  [112,] 0.375000000 0.3750000000 1.250000e-01
##  [113,] 0.426168977 0.1482326877 1.718640e-02
##  [114,] 0.321175019 0.4163379880 1.798991e-01
##  [115,] 0.443086838 0.2436977611 4.467792e-02
##  [116,] 0.362525595 0.3866939680 1.374912e-01
##  [117,] 0.444358195 0.2275981001 3.885821e-02
##  [118,] 0.335537578 0.4074384881 1.649156e-01
##  [119,] 0.362525595 0.3866939680 1.374912e-01
##  [120,] 0.386693968 0.3625255950 1.132892e-01
##  [121,] 0.397531973 0.3493462791 1.023338e-01
##  [122,] 0.444358195 0.2275981001 3.885821e-02
##  [123,] 0.424154946 0.3063341278 7.374710e-02
##  [124,] 0.442218287 0.1953987782 2.877966e-02
##  [125,] 0.335537578 0.4074384881 1.649156e-01
##  [126,] 0.293645732 0.0435030714 2.148300e-03
##  [127,] 0.392899701 0.1042386963 9.218388e-03
##  [128,] 0.243697761 0.4430868383 2.685375e-01
##  [129,] 0.377630828 0.0906313987 7.250512e-03
##  [130,] 0.318229499 0.0540389715 3.058810e-03
##  [131,] 0.443086838 0.2436977611 4.467792e-02
##  [132,] 0.291090430 0.4308138364 2.125348e-01
##  [133,] 0.433331375 0.1637029640 2.061445e-02
##  [134,] 0.360146521 0.0776786613 5.584740e-03
##  [135,] 0.266544426 0.0339238361 1.439193e-03
##  [136,] 0.443086838 0.2436977611 4.467792e-02
##  [137,] 0.318229499 0.0540389715 3.058810e-03
##  [138,] 0.375000000 0.3750000000 1.250000e-01
##  [139,] 0.169380014 0.0116813803 2.685375e-04
##  [140,] 0.375000000 0.3750000000 1.250000e-01
##  [141,] 0.266544426 0.0339238361 1.439193e-03
##  [142,] 0.360146521 0.0776786613 5.584740e-03
##  [143,] 0.442218287 0.1953987782 2.877966e-02
##  [144,] 0.433331375 0.1637029640 2.061445e-02
##  [145,] 0.243697761 0.4430868383 2.685375e-01
##  [146,] 0.444358195 0.2275981001 3.885821e-02
##  [147,] 0.440355309 0.2596967205 5.105149e-02
##  [148,] 0.442218287 0.1953987782 2.877966e-02
##  [149,] 0.179450170 0.4386559699 3.574234e-01
##  [150,] 0.318229499 0.0540389715 3.058810e-03
##  [151,] 0.442218287 0.1953987782 2.877966e-02
##  [152,] 0.275519452 0.4362391326 2.302373e-01
##  [153,] 0.438655970 0.1794501695 2.447048e-02
##  [154,] 0.204487093 0.0179374643 5.244873e-04
##  [155,] 0.407438488 0.3355375785 9.210835e-02
##  [156,] 0.293645732 0.0435030714 2.148300e-03
##  [157,] 0.430813836 0.2910904300 6.556091e-02
##  [158,] 0.438655970 0.1794501695 2.447048e-02
##  [159,] 0.306334128 0.4241549461 1.957638e-01
##  [160,] 0.443086838 0.2436977611 4.467792e-02
##  [161,] 0.426168977 0.1482326877 1.718640e-02
##  [162,] 0.430813836 0.2910904300 6.556091e-02
##  [163,] 0.227598100 0.4443581954 2.891855e-01
##  [164,] 0.211473264 0.4440938538 3.108657e-01
##  [165,] 0.375000000 0.3750000000 1.250000e-01
##  [166,] 0.416337988 0.3211750193 8.258786e-02
##  [167,] 0.426168977 0.1482326877 1.718640e-02
##  [168,] 0.169380014 0.0116813803 2.685375e-04
##  [169,] 0.443086838 0.2436977611 4.467792e-02
##  [170,] 0.440355309 0.2596967205 5.105149e-02
##  [171,] 0.438655970 0.1794501695 2.447048e-02
##  [172,] 0.397531973 0.3493462791 1.023338e-01
##  [173,] 0.433331375 0.1637029640 2.061445e-02
##  [174,] 0.443086838 0.2436977611 4.467792e-02
##  [175,] 0.259696720 0.4403553087 2.488965e-01
##  [176,] 0.033923836 0.2665444262 6.980925e-01
##  [177,] 0.360146521 0.0776786613 5.584740e-03
##  [178,] 0.377630828 0.0906313987 7.250512e-03
##  [179,] 0.360146521 0.0776786613 5.584740e-03
##  [180,] 0.438655970 0.1794501695 2.447048e-02
##  [181,] 0.444358195 0.2275981001 3.885821e-02
##  [182,] 0.386693968 0.3625255950 1.132892e-01
##  [183,] 0.416337988 0.3211750193 8.258786e-02
##  [184,] 0.362525595 0.3866939680 1.374912e-01
##  [185,] 0.243697761 0.4430868383 2.685375e-01
##  [186,] 0.386693968 0.3625255950 1.132892e-01
##  [187,] 0.440355309 0.2596967205 5.105149e-02
##  [188,] 0.318229499 0.0540389715 3.058810e-03
##  [189,] 0.424154946 0.3063341278 7.374710e-02
##  [190,] 0.406028666 0.1184250277 1.151354e-02
##  [191,] 0.407438488 0.3355375785 9.210835e-02
##  [192,] 0.169380014 0.0116813803 2.685375e-04
##  [193,] 0.321175019 0.4163379880 1.798991e-01
##  [194,] 0.426168977 0.1482326877 1.718640e-02
##  [195,] 0.444093854 0.2114732637 3.356718e-02
##  [196,] 0.266544426 0.0339238361 1.439193e-03
##  [197,] 0.360146521 0.0776786613 5.584740e-03
##  [198,] 0.340371253 0.0654560102 4.195898e-03
##  [199,] 0.291090430 0.4308138364 2.125348e-01
##  [200,] 0.275519452 0.4362391326 2.302373e-01
##  [201,] 0.195398778 0.4422182874 3.336033e-01
##  [202,] 0.397531973 0.3493462791 1.023338e-01
##  [203,] 0.335537578 0.4074384881 1.649156e-01
##  [204,] 0.417093250 0.1331148669 1.416116e-02
##  [205,] 0.243697761 0.4430868383 2.685375e-01
##  [206,] 0.318229499 0.0540389715 3.058810e-03
##  [207,] 0.335537578 0.4074384881 1.649156e-01
##  [208,] 0.416337988 0.3211750193 8.258786e-02
##  [209,] 0.169380014 0.0116813803 2.685375e-04
##  [210,] 0.266544426 0.0339238361 1.439193e-03
##  [211,] 0.438655970 0.1794501695 2.447048e-02
##  [212,] 0.392899701 0.1042386963 9.218388e-03
##  [213,] 0.335537578 0.4074384881 1.649156e-01
##  [214,] 0.407438488 0.3355375785 9.210835e-02
##  [215,] 0.416337988 0.3211750193 8.258786e-02
##  [216,] 0.443086838 0.2436977611 4.467792e-02
##  [217,] 0.436239133 0.2755194522 5.800410e-02
##  [218,] 0.440355309 0.2596967205 5.105149e-02
##  [219,] 0.266544426 0.0339238361 1.439193e-03
##  [220,] 0.236850055 0.0253767916 9.063140e-04
##  [221,] 0.349346279 0.3975319727 1.507880e-01
##  [222,] 0.440355309 0.2596967205 5.105149e-02
##  [223,] 0.377630828 0.0906313987 7.250512e-03
##  [224,] 0.291090430 0.4308138364 2.125348e-01
##  [225,] 0.204487093 0.0179374643 5.244873e-04
##  [226,] 0.211473264 0.4440938538 3.108657e-01
##  [227,] 0.443086838 0.2436977611 4.467792e-02
##  [228,] 0.000000000 0.0000000000 1.000000e+00
##  [229,] 0.443086838 0.2436977611 4.467792e-02
##  [230,] 0.433331375 0.1637029640 2.061445e-02
##  [231,] 0.291090430 0.4308138364 2.125348e-01
##  [232,] 0.236850055 0.0253767916 9.063140e-04
##  [233,] 0.444358195 0.2275981001 3.885821e-02
##  [234,] 0.377630828 0.0906313987 7.250512e-03
##  [235,] 0.090631399 0.3776308281 5.244873e-01
##  [236,] 0.306334128 0.4241549461 1.957638e-01
##  [237,] 0.318229499 0.0540389715 3.058810e-03
##  [238,] 0.426168977 0.1482326877 1.718640e-02
##  [239,] 0.321175019 0.4163379880 1.798991e-01
##  [240,] 0.227598100 0.4443581954 2.891855e-01
##  [241,] 0.416337988 0.3211750193 8.258786e-02
##  [242,] 0.430813836 0.2910904300 6.556091e-02
##  [243,] 0.377630828 0.0906313987 7.250512e-03
##  [244,] 0.436239133 0.2755194522 5.800410e-02
##  [245,] 0.204487093 0.0179374643 5.244873e-04
##  [246,] 0.243697761 0.4430868383 2.685375e-01
##  [247,] 0.417093250 0.1331148669 1.416116e-02
##  [248,] 0.275519452 0.4362391326 2.302373e-01
##  [249,] 0.442218287 0.1953987782 2.877966e-02
##  [250,] 0.417093250 0.1331148669 1.416116e-02
##  [251,] 0.362525595 0.3866939680 1.374912e-01
##  [252,] 0.430813836 0.2910904300 6.556091e-02
##  [253,] 0.321175019 0.4163379880 1.798991e-01
##  [254,] 0.442218287 0.1953987782 2.877966e-02
##  [255,] 0.090631399 0.0030210466 3.356718e-05
##  [256,] 0.293645732 0.0435030714 2.148300e-03
##  [257,] 0.360146521 0.0776786613 5.584740e-03
##  [258,] 0.259696720 0.4403553087 2.488965e-01
##  [259,] 0.397531973 0.3493462791 1.023338e-01
##  [260,] 0.444093854 0.2114732637 3.356718e-02
##  [261,] 0.204487093 0.0179374643 5.244873e-04
##  [262,] 0.392899701 0.1042386963 9.218388e-03
##  [263,] 0.430813836 0.2910904300 6.556091e-02
##  [264,] 0.417093250 0.1331148669 1.416116e-02
##  [265,] 0.386693968 0.3625255950 1.132892e-01
##  [266,] 0.377630828 0.0906313987 7.250512e-03
##  [267,] 0.424154946 0.3063341278 7.374710e-02
##  [268,] 0.444093854 0.2114732637 3.356718e-02
##  [269,] 0.397531973 0.3493462791 1.023338e-01
##  [270,] 0.340371253 0.0654560102 4.195898e-03
##  [271,] 0.204487093 0.0179374643 5.244873e-04
##  [272,] 0.318229499 0.0540389715 3.058810e-03
##  [273,] 0.417093250 0.1331148669 1.416116e-02
##  [274,] 0.375000000 0.3750000000 1.250000e-01
##  [275,] 0.318229499 0.0540389715 3.058810e-03
##  [276,] 0.386693968 0.3625255950 1.132892e-01
##  [277,] 0.444093854 0.2114732637 3.356718e-02
##  [278,] 0.243697761 0.4430868383 2.685375e-01
##  [279,] 0.407438488 0.3355375785 9.210835e-02
##  [280,] 0.321175019 0.4163379880 1.798991e-01
##  [281,] 0.436239133 0.2755194522 5.800410e-02
##  [282,] 0.443086838 0.2436977611 4.467792e-02
##  [283,] 0.433331375 0.1637029640 2.061445e-02
##  [284,] 0.362525595 0.3866939680 1.374912e-01
##  [285,] 0.426168977 0.1482326877 1.718640e-02
##  [286,] 0.386693968 0.3625255950 1.132892e-01
##  [287,] 0.375000000 0.3750000000 1.250000e-01
##  [288,] 0.440355309 0.2596967205 5.105149e-02
##  [289,] 0.243697761 0.4430868383 2.685375e-01
##  [290,] 0.362525595 0.3866939680 1.374912e-01
##  [291,] 0.444093854 0.2114732637 3.356718e-02
##  [292,] 0.377630828 0.0906313987 7.250512e-03
##  [293,] 0.424154946 0.3063341278 7.374710e-02
##  [294,] 0.243697761 0.4430868383 2.685375e-01
##  [295,] 0.416337988 0.3211750193 8.258786e-02
##  [296,] 0.424154946 0.3063341278 7.374710e-02
##  [297,] 0.416337988 0.3211750193 8.258786e-02
##  [298,] 0.349346279 0.3975319727 1.507880e-01
##  [299,] 0.195398778 0.4422182874 3.336033e-01
##  [300,] 0.416337988 0.3211750193 8.258786e-02
##  [301,] 0.426168977 0.1482326877 1.718640e-02
##  [302,] 0.417093250 0.1331148669 1.416116e-02
##  [303,] 0.362525595 0.3866939680 1.374912e-01
##  [304,] 0.362525595 0.3866939680 1.374912e-01
##  [305,] 0.444358195 0.2275981001 3.885821e-02
##  [306,] 0.293645732 0.0435030714 2.148300e-03
##  [307,] 0.375000000 0.3750000000 1.250000e-01
##  [308,] 0.392899701 0.1042386963 9.218388e-03
##  [309,] 0.275519452 0.4362391326 2.302373e-01
##  [310,] 0.195398778 0.4422182874 3.336033e-01
##  [311,] 0.275519452 0.4362391326 2.302373e-01
##  [312,] 0.349346279 0.3975319727 1.507880e-01
##  [313,] 0.436239133 0.2755194522 5.800410e-02
##  [314,] 0.416337988 0.3211750193 8.258786e-02
##  [315,] 0.386693968 0.3625255950 1.132892e-01
##  [316,] 0.417093250 0.1331148669 1.416116e-02
##  [317,] 0.392899701 0.1042386963 9.218388e-03
##  [318,] 0.386693968 0.3625255950 1.132892e-01
##  [319,] 0.433331375 0.1637029640 2.061445e-02
##  [320,] 0.397531973 0.3493462791 1.023338e-01
##  [321,] 0.416337988 0.3211750193 8.258786e-02
##  [322,] 0.407438488 0.3355375785 9.210835e-02
##  [323,] 0.397531973 0.3493462791 1.023338e-01
##  [324,] 0.375000000 0.3750000000 1.250000e-01
##  [325,] 0.438655970 0.1794501695 2.447048e-02
##  [326,] 0.349346279 0.3975319727 1.507880e-01
##  [327,] 0.407438488 0.3355375785 9.210835e-02
##  [328,] 0.430813836 0.2910904300 6.556091e-02
##  [329,] 0.424154946 0.3063341278 7.374710e-02
##  [330,] 0.195398778 0.4422182874 3.336033e-01
##  [331,] 0.442218287 0.1953987782 2.877966e-02
##  [332,] 0.444093854 0.2114732637 3.356718e-02
##  [333,] 0.440355309 0.2596967205 5.105149e-02
##  [334,] 0.377630828 0.0906313987 7.250512e-03
##  [335,] 0.349346279 0.3975319727 1.507880e-01
##  [336,] 0.433331375 0.1637029640 2.061445e-02
##  [337,] 0.318229499 0.0540389715 3.058810e-03
##  [338,] 0.349346279 0.3975319727 1.507880e-01
##  [339,] 0.440355309 0.2596967205 5.105149e-02
##  [340,] 0.163702964 0.4333313752 3.823512e-01
##  [341,] 0.340371253 0.0654560102 4.195898e-03
##  [342,] 0.362525595 0.3866939680 1.374912e-01
##  [343,] 0.440355309 0.2596967205 5.105149e-02
##  [344,] 0.204487093 0.0179374643 5.244873e-04
##  [345,] 0.416337988 0.3211750193 8.258786e-02
##  [346,] 0.163702964 0.4333313752 3.823512e-01
##  [347,] 0.227598100 0.4443581954 2.891855e-01
##  [348,] 0.377630828 0.0906313987 7.250512e-03
##  [349,] 0.416337988 0.3211750193 8.258786e-02
##  [350,] 0.335537578 0.4074384881 1.649156e-01
##  [351,] 0.306334128 0.4241549461 1.957638e-01
##  [352,] 0.377630828 0.0906313987 7.250512e-03
##  [353,] 0.397531973 0.3493462791 1.023338e-01
##  [354,] 0.397531973 0.3493462791 1.023338e-01
##  [355,] 0.444358195 0.2275981001 3.885821e-02
##  [356,] 0.362525595 0.3866939680 1.374912e-01
##  [357,] 0.397531973 0.3493462791 1.023338e-01
##  [358,] 0.416337988 0.3211750193 8.258786e-02
##  [359,] 0.424154946 0.3063341278 7.374710e-02
##  [360,] 0.436239133 0.2755194522 5.800410e-02
##  [361,] 0.275519452 0.4362391326 2.302373e-01
##  [362,] 0.362525595 0.3866939680 1.374912e-01
##  [363,] 0.321175019 0.4163379880 1.798991e-01
##  [364,] 0.444093854 0.2114732637 3.356718e-02
##  [365,] 0.275519452 0.4362391326 2.302373e-01
##  [366,] 0.362525595 0.3866939680 1.374912e-01
##  [367,] 0.375000000 0.3750000000 1.250000e-01
##  [368,] 0.436239133 0.2755194522 5.800410e-02
##  [369,] 0.362525595 0.3866939680 1.374912e-01
##  [370,] 0.321175019 0.4163379880 1.798991e-01
##  [371,] 0.340371253 0.0654560102 4.195898e-03
##  [372,] 0.416337988 0.3211750193 8.258786e-02
##  [373,] 0.236850055 0.0253767916 9.063140e-04
##  [374,] 0.266544426 0.0339238361 1.439193e-03
##  [375,] 0.397531973 0.3493462791 1.023338e-01
##  [376,] 0.444093854 0.2114732637 3.356718e-02
##  [377,] 0.417093250 0.1331148669 1.416116e-02
##  [378,] 0.444358195 0.2275981001 3.885821e-02
##  [379,] 0.407438488 0.3355375785 9.210835e-02
##  [380,] 0.195398778 0.4422182874 3.336033e-01
##  [381,] 0.406028666 0.1184250277 1.151354e-02
##  [382,] 0.195398778 0.4422182874 3.336033e-01
##  [383,] 0.416337988 0.3211750193 8.258786e-02
##  [384,] 0.243697761 0.4430868383 2.685375e-01
##  [385,] 0.266544426 0.0339238361 1.439193e-03
##  [386,] 0.426168977 0.1482326877 1.718640e-02
##  [387,] 0.424154946 0.3063341278 7.374710e-02
##  [388,] 0.148232688 0.4261689772 4.084119e-01
##  [389,] 0.306334128 0.4241549461 1.957638e-01
##  [390,] 0.436239133 0.2755194522 5.800410e-02
##  [391,] 0.392899701 0.1042386963 9.218388e-03
##  [392,] 0.266544426 0.0339238361 1.439193e-03
##  [393,] 0.349346279 0.3975319727 1.507880e-01
##  [394,] 0.340371253 0.0654560102 4.195898e-03
##  [395,] 0.321175019 0.4163379880 1.798991e-01
##  [396,] 0.407438488 0.3355375785 9.210835e-02
##  [397,] 0.444093854 0.2114732637 3.356718e-02
##  [398,] 0.444358195 0.2275981001 3.885821e-02
##  [399,] 0.442218287 0.1953987782 2.877966e-02
##  [400,] 0.227598100 0.4443581954 2.891855e-01
##  [401,] 0.417093250 0.1331148669 1.416116e-02
##  [402,] 0.204487093 0.0179374643 5.244873e-04
##  [403,] 0.442218287 0.1953987782 2.877966e-02
##  [404,] 0.318229499 0.0540389715 3.058810e-03
##  [405,] 0.397531973 0.3493462791 1.023338e-01
##  [406,] 0.335537578 0.4074384881 1.649156e-01
##  [407,] 0.442218287 0.1953987782 2.877966e-02
##  [408,] 0.426168977 0.1482326877 1.718640e-02
##  [409,] 0.349346279 0.3975319727 1.507880e-01
##  [410,] 0.362525595 0.3866939680 1.374912e-01
##  [411,] 0.306334128 0.4241549461 1.957638e-01
##  [412,] 0.362525595 0.3866939680 1.374912e-01
##  [413,] 0.406028666 0.1184250277 1.151354e-02
##  [414,] 0.442218287 0.1953987782 2.877966e-02
##  [415,] 0.046838810 0.0007678494 4.195898e-06
##  [416,] 0.406028666 0.1184250277 1.151354e-02
##  [417,] 0.436239133 0.2755194522 5.800410e-02
##  [418,] 0.430813836 0.2910904300 6.556091e-02
##  [419,] 0.424154946 0.3063341278 7.374710e-02
##  [420,] 0.443086838 0.2436977611 4.467792e-02
##  [421,] 0.430813836 0.2910904300 6.556091e-02
##  [422,] 0.406028666 0.1184250277 1.151354e-02
##  [423,] 0.195398778 0.4422182874 3.336033e-01
##  [424,] 0.397531973 0.3493462791 1.023338e-01
##  [425,] 0.291090430 0.4308138364 2.125348e-01
##  [426,] 0.335537578 0.4074384881 1.649156e-01
##  [427,] 0.318229499 0.0540389715 3.058810e-03
##  [428,] 0.169380014 0.0116813803 2.685375e-04
##  [429,] 0.436239133 0.2755194522 5.800410e-02
##  [430,] 0.392899701 0.1042386963 9.218388e-03
##  [431,] 0.227598100 0.4443581954 2.891855e-01
##  [432,] 0.438655970 0.1794501695 2.447048e-02
##  [433,] 0.406028666 0.1184250277 1.151354e-02
##  [434,] 0.406028666 0.1184250277 1.151354e-02
##  [435,] 0.266544426 0.0339238361 1.439193e-03
##  [436,] 0.430813836 0.2910904300 6.556091e-02
##  [437,] 0.424154946 0.3063341278 7.374710e-02
##  [438,] 0.259696720 0.4403553087 2.488965e-01
##  [439,] 0.440355309 0.2596967205 5.105149e-02
##  [440,] 0.444093854 0.2114732637 3.356718e-02
##  [441,] 0.243697761 0.4430868383 2.685375e-01
##  [442,] 0.227598100 0.4443581954 2.891855e-01
##  [443,] 0.444358195 0.2275981001 3.885821e-02
##  [444,] 0.424154946 0.3063341278 7.374710e-02
##  [445,] 0.065456010 0.3403712531 5.899768e-01
##  [446,] 0.318229499 0.0540389715 3.058810e-03
##  [447,] 0.397531973 0.3493462791 1.023338e-01
##  [448,] 0.360146521 0.0776786613 5.584740e-03
##  [449,] 0.436239133 0.2755194522 5.800410e-02
##  [450,] 0.349346279 0.3975319727 1.507880e-01
##  [451,] 0.444358195 0.2275981001 3.885821e-02
##  [452,] 0.204487093 0.0179374643 5.244873e-04
##  [453,] 0.392899701 0.1042386963 9.218388e-03
##  [454,] 0.227598100 0.4443581954 2.891855e-01
##  [455,] 0.436239133 0.2755194522 5.800410e-02
##  [456,] 0.433331375 0.1637029640 2.061445e-02
##  [457,] 0.444093854 0.2114732637 3.356718e-02
##  [458,] 0.416337988 0.3211750193 8.258786e-02
##  [459,] 0.243697761 0.4430868383 2.685375e-01
##  [460,] 0.293645732 0.0435030714 2.148300e-03
##  [461,] 0.377630828 0.0906313987 7.250512e-03
##  [462,] 0.306334128 0.4241549461 1.957638e-01
##  [463,] 0.335537578 0.4074384881 1.649156e-01
##  [464,] 0.033923836 0.2665444262 6.980925e-01
##  [465,] 0.133114867 0.4170932496 4.356307e-01
##  [466,] 0.321175019 0.4163379880 1.798991e-01
##  [467,] 0.335537578 0.4074384881 1.649156e-01
##  [468,] 0.259696720 0.4403553087 2.488965e-01
##  [469,] 0.406028666 0.1184250277 1.151354e-02
##  [470,] 0.349346279 0.3975319727 1.507880e-01
##  [471,] 0.430813836 0.2910904300 6.556091e-02
##  [472,] 0.362525595 0.3866939680 1.374912e-01
##  [473,] 0.321175019 0.4163379880 1.798991e-01
##  [474,] 0.306334128 0.4241549461 1.957638e-01
##  [475,] 0.443086838 0.2436977611 4.467792e-02
##  [476,] 0.377630828 0.0906313987 7.250512e-03
##  [477,] 0.416337988 0.3211750193 8.258786e-02
##  [478,] 0.291090430 0.4308138364 2.125348e-01
##  [479,] 0.416337988 0.3211750193 8.258786e-02
##  [480,] 0.424154946 0.3063341278 7.374710e-02
##  [481,] 0.442218287 0.1953987782 2.877966e-02
##  [482,] 0.440355309 0.2596967205 5.105149e-02
##  [483,] 0.335537578 0.4074384881 1.649156e-01
##  [484,] 0.291090430 0.4308138364 2.125348e-01
##  [485,] 0.430813836 0.2910904300 6.556091e-02
##  [486,] 0.318229499 0.0540389715 3.058810e-03
##  [487,] 0.430813836 0.2910904300 6.556091e-02
##  [488,] 0.407438488 0.3355375785 9.210835e-02
##  [489,] 0.386693968 0.3625255950 1.132892e-01
##  [490,] 0.360146521 0.0776786613 5.584740e-03
##  [491,] 0.236850055 0.0253767916 9.063140e-04
##  [492,] 0.362525595 0.3866939680 1.374912e-01
##  [493,] 0.236850055 0.0253767916 9.063140e-04
##  [494,] 0.436239133 0.2755194522 5.800410e-02
##  [495,] 0.375000000 0.3750000000 1.250000e-01
##  [496,] 0.443086838 0.2436977611 4.467792e-02
##  [497,] 0.440355309 0.2596967205 5.105149e-02
##  [498,] 0.426168977 0.1482326877 1.718640e-02
##  [499,] 0.236850055 0.0253767916 9.063140e-04
##  [500,] 0.424154946 0.3063341278 7.374710e-02
##  [501,] 0.266544426 0.0339238361 1.439193e-03
##  [502,] 0.443086838 0.2436977611 4.467792e-02
##  [503,] 0.266544426 0.0339238361 1.439193e-03
##  [504,] 0.424154946 0.3063341278 7.374710e-02
##  [505,] 0.243697761 0.4430868383 2.685375e-01
##  [506,] 0.335537578 0.4074384881 1.649156e-01
##  [507,] 0.211473264 0.4440938538 3.108657e-01
##  [508,] 0.349346279 0.3975319727 1.507880e-01
##  [509,] 0.416337988 0.3211750193 8.258786e-02
##  [510,] 0.430813836 0.2910904300 6.556091e-02
##  [511,] 0.416337988 0.3211750193 8.258786e-02
##  [512,] 0.443086838 0.2436977611 4.467792e-02
##  [513,] 0.349346279 0.3975319727 1.507880e-01
##  [514,] 0.335537578 0.4074384881 1.649156e-01
##  [515,] 0.392899701 0.1042386963 9.218388e-03
##  [516,] 0.443086838 0.2436977611 4.467792e-02
##  [517,] 0.293645732 0.0435030714 2.148300e-03
##  [518,] 0.375000000 0.3750000000 1.250000e-01
##  [519,] 0.444093854 0.2114732637 3.356718e-02
##  [520,] 0.362525595 0.3866939680 1.374912e-01
##  [521,] 0.360146521 0.0776786613 5.584740e-03
##  [522,] 0.417093250 0.1331148669 1.416116e-02
##  [523,] 0.179450170 0.4386559699 3.574234e-01
##  [524,] 0.416337988 0.3211750193 8.258786e-02
##  [525,] 0.275519452 0.4362391326 2.302373e-01
##  [526,] 0.243697761 0.4430868383 2.685375e-01
##  [527,] 0.444358195 0.2275981001 3.885821e-02
##  [528,] 0.375000000 0.3750000000 1.250000e-01
##  [529,] 0.236850055 0.0253767916 9.063140e-04
##  [530,] 0.243697761 0.4430868383 2.685375e-01
##  [531,] 0.397531973 0.3493462791 1.023338e-01
##  [532,] 0.440355309 0.2596967205 5.105149e-02
##  [533,] 0.054038972 0.3182294988 6.246727e-01
##  [534,] 0.397531973 0.3493462791 1.023338e-01
##  [535,] 0.444093854 0.2114732637 3.356718e-02
##  [536,] 0.392899701 0.1042386963 9.218388e-03
##  [537,] 0.275519452 0.4362391326 2.302373e-01
##  [538,] 0.424154946 0.3063341278 7.374710e-02
##  [539,] 0.417093250 0.1331148669 1.416116e-02
##  [540,] 0.392899701 0.1042386963 9.218388e-03
##  [541,] 0.291090430 0.4308138364 2.125348e-01
##  [542,] 0.386693968 0.3625255950 1.132892e-01
##  [543,] 0.291090430 0.4308138364 2.125348e-01
##  [544,] 0.407438488 0.3355375785 9.210835e-02
##  [545,] 0.386693968 0.3625255950 1.132892e-01
##  [546,] 0.204487093 0.0179374643 5.244873e-04
##  [547,] 0.211473264 0.4440938538 3.108657e-01
##  [548,] 0.426168977 0.1482326877 1.718640e-02
##  [549,] 0.416337988 0.3211750193 8.258786e-02
##  [550,] 0.340371253 0.0654560102 4.195898e-03
##  [551,] 0.417093250 0.1331148669 1.416116e-02
##  [552,] 0.243697761 0.4430868383 2.685375e-01
##  [553,] 0.397531973 0.3493462791 1.023338e-01
##  [554,] 0.236850055 0.0253767916 9.063140e-04
##  [555,] 0.275519452 0.4362391326 2.302373e-01
##  [556,] 0.275519452 0.4362391326 2.302373e-01
##  [557,] 0.204487093 0.0179374643 5.244873e-04
##  [558,] 0.416337988 0.3211750193 8.258786e-02
##  [559,] 0.243697761 0.4430868383 2.685375e-01
##  [560,] 0.377630828 0.0906313987 7.250512e-03
##  [561,] 0.386693968 0.3625255950 1.132892e-01
##  [562,] 0.442218287 0.1953987782 2.877966e-02
##  [563,] 0.375000000 0.3750000000 1.250000e-01
##  [564,] 0.392899701 0.1042386963 9.218388e-03
##  [565,] 0.335537578 0.4074384881 1.649156e-01
##  [566,] 0.065456010 0.3403712531 5.899768e-01
##  [567,] 0.426168977 0.1482326877 1.718640e-02
##  [568,] 0.444093854 0.2114732637 3.356718e-02
##  [569,] 0.340371253 0.0654560102 4.195898e-03
##  [570,] 0.444093854 0.2114732637 3.356718e-02
##  [571,] 0.444358195 0.2275981001 3.885821e-02
##  [572,] 0.335537578 0.4074384881 1.649156e-01
##  [573,] 0.426168977 0.1482326877 1.718640e-02
##  [574,] 0.417093250 0.1331148669 1.416116e-02
##  [575,] 0.243697761 0.4430868383 2.685375e-01
##  [576,] 0.444093854 0.2114732637 3.356718e-02
##  [577,] 0.444093854 0.2114732637 3.356718e-02
##  [578,] 0.392899701 0.1042386963 9.218388e-03
##  [579,] 0.321175019 0.4163379880 1.798991e-01
##  [580,] 0.131453291 0.0066840657 1.132892e-04
##  [581,] 0.444093854 0.2114732637 3.356718e-02
##  [582,] 0.340371253 0.0654560102 4.195898e-03
##  [583,] 0.406028666 0.1184250277 1.151354e-02
##  [584,] 0.340371253 0.0654560102 4.195898e-03
##  [585,] 0.436239133 0.2755194522 5.800410e-02
##  [586,] 0.340371253 0.0654560102 4.195898e-03
##  [587,] 0.386693968 0.3625255950 1.132892e-01
##  [588,] 0.291090430 0.4308138364 2.125348e-01
##  [589,] 0.442218287 0.1953987782 2.877966e-02
##  [590,] 0.090631399 0.3776308281 5.244873e-01
##  [591,] 0.133114867 0.4170932496 4.356307e-01
##  [592,] 0.442218287 0.1953987782 2.877966e-02
##  [593,] 0.417093250 0.1331148669 1.416116e-02
##  [594,] 0.046838810 0.0007678494 4.195898e-06
##  [595,] 0.362525595 0.3866939680 1.374912e-01
##  [596,] 0.443086838 0.2436977611 4.467792e-02
##  [597,] 0.118425028 0.4060286664 4.640328e-01
##  [598,] 0.433331375 0.1637029640 2.061445e-02
##  [599,] 0.417093250 0.1331148669 1.416116e-02
##  [600,] 0.424154946 0.3063341278 7.374710e-02
##  [601,] 0.397531973 0.3493462791 1.023338e-01
##  [602,] 0.291090430 0.4308138364 2.125348e-01
##  [603,] 0.417093250 0.1331148669 1.416116e-02
##  [604,] 0.275519452 0.4362391326 2.302373e-01
##  [605,] 0.397531973 0.3493462791 1.023338e-01
##  [606,] 0.416337988 0.3211750193 8.258786e-02
##  [607,] 0.424154946 0.3063341278 7.374710e-02
##  [608,] 0.266544426 0.0339238361 1.439193e-03
##  [609,] 0.416337988 0.3211750193 8.258786e-02
##  [610,] 0.275519452 0.4362391326 2.302373e-01
##  [611,] 0.397531973 0.3493462791 1.023338e-01
##  [612,] 0.444358195 0.2275981001 3.885821e-02
##  [613,] 0.386693968 0.3625255950 1.132892e-01
##  [614,] 0.436239133 0.2755194522 5.800410e-02
##  [615,] 0.291090430 0.4308138364 2.125348e-01
##  [616,] 0.195398778 0.4422182874 3.336033e-01
##  [617,] 0.444358195 0.2275981001 3.885821e-02
##  [618,] 0.377630828 0.0906313987 7.250512e-03
##  [619,] 0.375000000 0.3750000000 1.250000e-01
##  [620,] 0.417093250 0.1331148669 1.416116e-02
##  [621,] 0.392899701 0.1042386963 9.218388e-03
##  [622,] 0.291090430 0.4308138364 2.125348e-01
##  [623,] 0.438655970 0.1794501695 2.447048e-02
##  [624,] 0.417093250 0.1331148669 1.416116e-02
##  [625,] 0.386693968 0.3625255950 1.132892e-01
##  [626,] 0.211473264 0.4440938538 3.108657e-01
##  [627,] 0.340371253 0.0654560102 4.195898e-03
##  [628,] 0.360146521 0.0776786613 5.584740e-03
##  [629,] 0.406028666 0.1184250277 1.151354e-02
##  [630,] 0.417093250 0.1331148669 1.416116e-02
##  [631,] 0.443086838 0.2436977611 4.467792e-02
##  [632,] 0.436239133 0.2755194522 5.800410e-02
##  [633,] 0.444358195 0.2275981001 3.885821e-02
##  [634,] 0.424154946 0.3063341278 7.374710e-02
##  [635,] 0.430813836 0.2910904300 6.556091e-02
##  [636,] 0.424154946 0.3063341278 7.374710e-02
##  [637,] 0.360146521 0.0776786613 5.584740e-03
##  [638,] 0.397531973 0.3493462791 1.023338e-01
##  [639,] 0.407438488 0.3355375785 9.210835e-02
##  [640,] 0.335537578 0.4074384881 1.649156e-01
##  [641,] 0.444093854 0.2114732637 3.356718e-02
##  [642,] 0.436239133 0.2755194522 5.800410e-02
##  [643,] 0.275519452 0.4362391326 2.302373e-01
##  [644,] 0.360146521 0.0776786613 5.584740e-03
##  [645,] 0.417093250 0.1331148669 1.416116e-02
##  [646,] 0.417093250 0.1331148669 1.416116e-02
##  [647,] 0.440355309 0.2596967205 5.105149e-02
##  [648,] 0.424154946 0.3063341278 7.374710e-02
##  [649,] 0.416337988 0.3211750193 8.258786e-02
##  [650,] 0.243697761 0.4430868383 2.685375e-01
##  [651,] 0.360146521 0.0776786613 5.584740e-03
##  [652,] 0.436239133 0.2755194522 5.800410e-02
##  [653,] 0.397531973 0.3493462791 1.023338e-01
##  [654,] 0.377630828 0.0906313987 7.250512e-03
##  [655,] 0.444358195 0.2275981001 3.885821e-02
##  [656,] 0.375000000 0.3750000000 1.250000e-01
##  [657,] 0.424154946 0.3063341278 7.374710e-02
##  [658,] 0.306334128 0.4241549461 1.957638e-01
##  [659,] 0.436239133 0.2755194522 5.800410e-02
##  [660,] 0.444358195 0.2275981001 3.885821e-02
##  [661,] 0.377630828 0.0906313987 7.250512e-03
##  [662,] 0.417093250 0.1331148669 1.416116e-02
##  [663,] 0.444093854 0.2114732637 3.356718e-02
##  [664,] 0.335537578 0.4074384881 1.649156e-01
##  [665,] 0.306334128 0.4241549461 1.957638e-01
##  [666,] 0.179450170 0.4386559699 3.574234e-01
##  [667,] 0.259696720 0.4403553087 2.488965e-01
##  [668,] 0.406028666 0.1184250277 1.151354e-02
##  [669,] 0.443086838 0.2436977611 4.467792e-02
##  [670,] 0.375000000 0.3750000000 1.250000e-01
##  [671,] 0.306334128 0.4241549461 1.957638e-01
##  [672,] 0.386693968 0.3625255950 1.132892e-01
##  [673,] 0.407438488 0.3355375785 9.210835e-02
##  [674,] 0.377630828 0.0906313987 7.250512e-03
##  [675,] 0.318229499 0.0540389715 3.058810e-03
##  [676,] 0.291090430 0.4308138364 2.125348e-01
##  [677,] 0.406028666 0.1184250277 1.151354e-02
##  [678,] 0.375000000 0.3750000000 1.250000e-01
##  [679,] 0.362525595 0.3866939680 1.374912e-01
##  [680,] 0.362525595 0.3866939680 1.374912e-01
##  [681,] 0.424154946 0.3063341278 7.374710e-02
##  [682,] 0.259696720 0.4403553087 2.488965e-01
##  [683,] 0.043503071 0.2936457319 6.607029e-01
##  [684,] 0.204487093 0.0179374643 5.244873e-04
##  [685,] 0.392899701 0.1042386963 9.218388e-03
##  [686,] 0.407438488 0.3355375785 9.210835e-02
##  [687,] 0.291090430 0.4308138364 2.125348e-01
##  [688,] 0.424154946 0.3063341278 7.374710e-02
##  [689,] 0.424154946 0.3063341278 7.374710e-02
##  [690,] 0.406028666 0.1184250277 1.151354e-02
##  [691,] 0.211473264 0.4440938538 3.108657e-01
##  [692,] 0.386693968 0.3625255950 1.132892e-01
##  [693,] 0.306334128 0.4241549461 1.957638e-01
##  [694,] 0.360146521 0.0776786613 5.584740e-03
##  [695,] 0.433331375 0.1637029640 2.061445e-02
##  [696,] 0.266544426 0.0339238361 1.439193e-03
##  [697,] 0.349346279 0.3975319727 1.507880e-01
##  [698,] 0.417093250 0.1331148669 1.416116e-02
##  [699,] 0.227598100 0.4443581954 2.891855e-01
##  [700,] 0.179450170 0.4386559699 3.574234e-01
##  [701,] 0.340371253 0.0654560102 4.195898e-03
##  [702,] 0.335537578 0.4074384881 1.649156e-01
##  [703,] 0.360146521 0.0776786613 5.584740e-03
##  [704,] 0.426168977 0.1482326877 1.718640e-02
##  [705,] 0.266544426 0.0339238361 1.439193e-03
##  [706,] 0.118425028 0.4060286664 4.640328e-01
##  [707,] 0.430813836 0.2910904300 6.556091e-02
##  [708,] 0.416337988 0.3211750193 8.258786e-02
##  [709,] 0.433331375 0.1637029640 2.061445e-02
##  [710,] 0.375000000 0.3750000000 1.250000e-01
##  [711,] 0.211473264 0.4440938538 3.108657e-01
##  [712,] 0.291090430 0.4308138364 2.125348e-01
##  [713,] 0.406028666 0.1184250277 1.151354e-02
##  [714,] 0.321175019 0.4163379880 1.798991e-01
##  [715,] 0.259696720 0.4403553087 2.488965e-01
##  [716,] 0.349346279 0.3975319727 1.507880e-01
##  [717,] 0.275519452 0.4362391326 2.302373e-01
##  [718,] 0.377630828 0.0906313987 7.250512e-03
##  [719,] 0.131453291 0.0066840657 1.132892e-04
##  [720,] 0.211473264 0.4440938538 3.108657e-01
##  [721,] 0.211473264 0.4440938538 3.108657e-01
##  [722,] 0.386693968 0.3625255950 1.132892e-01
##  [723,] 0.444358195 0.2275981001 3.885821e-02
##  [724,] 0.406028666 0.1184250277 1.151354e-02
##  [725,] 0.349346279 0.3975319727 1.507880e-01
##  [726,] 0.424154946 0.3063341278 7.374710e-02
##  [727,] 0.407438488 0.3355375785 9.210835e-02
##  [728,] 0.236850055 0.0253767916 9.063140e-04
##  [729,] 0.442218287 0.1953987782 2.877966e-02
##  [730,] 0.043503071 0.2936457319 6.607029e-01
##  [731,] 0.362525595 0.3866939680 1.374912e-01
##  [732,] 0.318229499 0.0540389715 3.058810e-03
##  [733,] 0.440355309 0.2596967205 5.105149e-02
##  [734,] 0.090631399 0.0030210466 3.356718e-05
##  [735,] 0.375000000 0.3750000000 1.250000e-01
##  [736,] 0.266544426 0.0339238361 1.439193e-03
##  [737,] 0.321175019 0.4163379880 1.798991e-01
##  [738,] 0.416337988 0.3211750193 8.258786e-02
##  [739,] 0.406028666 0.1184250277 1.151354e-02
##  [740,] 0.397531973 0.3493462791 1.023338e-01
##  [741,] 0.293645732 0.0435030714 2.148300e-03
##  [742,] 0.392899701 0.1042386963 9.218388e-03
##  [743,] 0.406028666 0.1184250277 1.151354e-02
##  [744,] 0.362525595 0.3866939680 1.374912e-01
##  [745,] 0.375000000 0.3750000000 1.250000e-01
##  [746,] 0.266544426 0.0339238361 1.439193e-03
##  [747,] 0.211473264 0.4440938538 3.108657e-01
##  [748,] 0.179450170 0.4386559699 3.574234e-01
##  [749,] 0.163702964 0.4333313752 3.823512e-01
##  [750,] 0.360146521 0.0776786613 5.584740e-03
##  [751,] 0.349346279 0.3975319727 1.507880e-01
##  [752,] 0.340371253 0.0654560102 4.195898e-03
##  [753,] 0.438655970 0.1794501695 2.447048e-02
##  [754,] 0.340371253 0.0654560102 4.195898e-03
##  [755,] 0.444093854 0.2114732637 3.356718e-02
##  [756,] 0.433331375 0.1637029640 2.061445e-02
##  [757,] 0.407438488 0.3355375785 9.210835e-02
##  [758,] 0.442218287 0.1953987782 2.877966e-02
##  [759,] 0.227598100 0.4443581954 2.891855e-01
##  [760,] 0.349346279 0.3975319727 1.507880e-01
##  [761,] 0.293645732 0.0435030714 2.148300e-03
##  [762,] 0.406028666 0.1184250277 1.151354e-02
##  [763,] 0.204487093 0.0179374643 5.244873e-04
##  [764,] 0.362525595 0.3866939680 1.374912e-01
##  [765,] 0.266544426 0.0339238361 1.439193e-03
##  [766,] 0.430813836 0.2910904300 6.556091e-02
##  [767,] 0.438655970 0.1794501695 2.447048e-02
##  [768,] 0.362525595 0.3866939680 1.374912e-01
##  [769,] 0.426168977 0.1482326877 1.718640e-02
##  [770,] 0.426168977 0.1482326877 1.718640e-02
##  [771,] 0.444358195 0.2275981001 3.885821e-02
##  [772,] 0.443086838 0.2436977611 4.467792e-02
##  [773,] 0.406028666 0.1184250277 1.151354e-02
##  [774,] 0.163702964 0.4333313752 3.823512e-01
##  [775,] 0.104238696 0.3928997013 4.936432e-01
##  [776,] 0.444358195 0.2275981001 3.885821e-02
##  [777,] 0.392899701 0.1042386963 9.218388e-03
##  [778,] 0.195398778 0.4422182874 3.336033e-01
##  [779,] 0.131453291 0.0066840657 1.132892e-04
##  [780,] 0.321175019 0.4163379880 1.798991e-01
##  [781,] 0.436239133 0.2755194522 5.800410e-02
##  [782,] 0.306334128 0.4241549461 1.957638e-01
##  [783,] 0.438655970 0.1794501695 2.447048e-02
##  [784,] 0.211473264 0.4440938538 3.108657e-01
##  [785,] 0.436239133 0.2755194522 5.800410e-02
##  [786,] 0.440355309 0.2596967205 5.105149e-02
##  [787,] 0.426168977 0.1482326877 1.718640e-02
##  [788,] 0.169380014 0.0116813803 2.685375e-04
##  [789,] 0.397531973 0.3493462791 1.023338e-01
##  [790,] 0.227598100 0.4443581954 2.891855e-01
##  [791,] 0.360146521 0.0776786613 5.584740e-03
##  [792,] 0.406028666 0.1184250277 1.151354e-02
##  [793,] 0.375000000 0.3750000000 1.250000e-01
##  [794,] 0.417093250 0.1331148669 1.416116e-02
##  [795,] 0.349346279 0.3975319727 1.507880e-01
##  [796,] 0.442218287 0.1953987782 2.877966e-02
##  [797,] 0.163702964 0.4333313752 3.823512e-01
##  [798,] 0.443086838 0.2436977611 4.467792e-02
##  [799,] 0.416337988 0.3211750193 8.258786e-02
##  [800,] 0.133114867 0.4170932496 4.356307e-01
##  [801,] 0.362525595 0.3866939680 1.374912e-01
##  [802,] 0.386693968 0.3625255950 1.132892e-01
##  [803,] 0.377630828 0.0906313987 7.250512e-03
##  [804,] 0.442218287 0.1953987782 2.877966e-02
##  [805,] 0.349346279 0.3975319727 1.507880e-01
##  [806,] 0.291090430 0.4308138364 2.125348e-01
##  [807,] 0.417093250 0.1331148669 1.416116e-02
##  [808,] 0.426168977 0.1482326877 1.718640e-02
##  [809,] 0.375000000 0.3750000000 1.250000e-01
##  [810,] 0.179450170 0.4386559699 3.574234e-01
##  [811,] 0.392899701 0.1042386963 9.218388e-03
##  [812,] 0.430813836 0.2910904300 6.556091e-02
##  [813,] 0.430813836 0.2910904300 6.556091e-02
##  [814,] 0.386693968 0.3625255950 1.132892e-01
##  [815,] 0.386693968 0.3625255950 1.132892e-01
##  [816,] 0.360146521 0.0776786613 5.584740e-03
##  [817,] 0.335537578 0.4074384881 1.649156e-01
##  [818,] 0.443086838 0.2436977611 4.467792e-02
##  [819,] 0.306334128 0.4241549461 1.957638e-01
##  [820,] 0.444093854 0.2114732637 3.356718e-02
##  [821,] 0.340371253 0.0654560102 4.195898e-03
##  [822,] 0.417093250 0.1331148669 1.416116e-02
##  [823,] 0.424154946 0.3063341278 7.374710e-02
##  [824,] 0.440355309 0.2596967205 5.105149e-02
##  [825,] 0.392899701 0.1042386963 9.218388e-03
##  [826,] 0.236850055 0.0253767916 9.063140e-04
##  [827,] 0.426168977 0.1482326877 1.718640e-02
##  [828,] 0.340371253 0.0654560102 4.195898e-03
##  [829,] 0.377630828 0.0906313987 7.250512e-03
##  [830,] 0.416337988 0.3211750193 8.258786e-02
##  [831,] 0.433331375 0.1637029640 2.061445e-02
##  [832,] 0.397531973 0.3493462791 1.023338e-01
##  [833,] 0.054038972 0.3182294988 6.246727e-01
##  [834,] 0.444358195 0.2275981001 3.885821e-02
##  [835,] 0.440355309 0.2596967205 5.105149e-02
##  [836,] 0.090631399 0.0030210466 3.356718e-05
##  [837,] 0.426168977 0.1482326877 1.718640e-02
##  [838,] 0.293645732 0.0435030714 2.148300e-03
##  [839,] 0.349346279 0.3975319727 1.507880e-01
##  [840,] 0.266544426 0.0339238361 1.439193e-03
##  [841,] 0.442218287 0.1953987782 2.877966e-02
##  [842,] 0.291090430 0.4308138364 2.125348e-01
##  [843,] 0.444358195 0.2275981001 3.885821e-02
##  [844,] 0.407438488 0.3355375785 9.210835e-02
##  [845,] 0.386693968 0.3625255950 1.132892e-01
##  [846,] 0.306334128 0.4241549461 1.957638e-01
##  [847,] 0.386693968 0.3625255950 1.132892e-01
##  [848,] 0.397531973 0.3493462791 1.023338e-01
##  [849,] 0.090631399 0.0030210466 3.356718e-05
##  [850,] 0.442218287 0.1953987782 2.877966e-02
##  [851,] 0.407438488 0.3355375785 9.210835e-02
##  [852,] 0.306334128 0.4241549461 1.957638e-01
##  [853,] 0.349346279 0.3975319727 1.507880e-01
##  [854,] 0.406028666 0.1184250277 1.151354e-02
##  [855,] 0.433331375 0.1637029640 2.061445e-02
##  [856,] 0.179450170 0.4386559699 3.574234e-01
##  [857,] 0.397531973 0.3493462791 1.023338e-01
##  [858,] 0.340371253 0.0654560102 4.195898e-03
##  [859,] 0.195398778 0.4422182874 3.336033e-01
##  [860,] 0.293645732 0.0435030714 2.148300e-03
##  [861,] 0.436239133 0.2755194522 5.800410e-02
##  [862,] 0.392899701 0.1042386963 9.218388e-03
##  [863,] 0.424154946 0.3063341278 7.374710e-02
##  [864,] 0.407438488 0.3355375785 9.210835e-02
##  [865,] 0.306334128 0.4241549461 1.957638e-01
##  [866,] 0.443086838 0.2436977611 4.467792e-02
##  [867,] 0.444093854 0.2114732637 3.356718e-02
##  [868,] 0.430813836 0.2910904300 6.556091e-02
##  [869,] 0.377630828 0.0906313987 7.250512e-03
##  [870,] 0.243697761 0.4430868383 2.685375e-01
##  [871,] 0.416337988 0.3211750193 8.258786e-02
##  [872,] 0.397531973 0.3493462791 1.023338e-01
##  [873,] 0.397531973 0.3493462791 1.023338e-01
##  [874,] 0.227598100 0.4443581954 2.891855e-01
##  [875,] 0.443086838 0.2436977611 4.467792e-02
##  [876,] 0.436239133 0.2755194522 5.800410e-02
##  [877,] 0.360146521 0.0776786613 5.584740e-03
##  [878,] 0.243697761 0.4430868383 2.685375e-01
##  [879,] 0.433331375 0.1637029640 2.061445e-02
##  [880,] 0.386693968 0.3625255950 1.132892e-01
##  [881,] 0.318229499 0.0540389715 3.058810e-03
##  [882,] 0.443086838 0.2436977611 4.467792e-02
##  [883,] 0.426168977 0.1482326877 1.718640e-02
##  [884,] 0.090631399 0.0030210466 3.356718e-05
##  [885,] 0.362525595 0.3866939680 1.374912e-01
##  [886,] 0.436239133 0.2755194522 5.800410e-02
##  [887,] 0.416337988 0.3211750193 8.258786e-02
##  [888,] 0.227598100 0.4443581954 2.891855e-01
##  [889,] 0.104238696 0.3928997013 4.936432e-01
##  [890,] 0.293645732 0.0435030714 2.148300e-03
##  [891,] 0.426168977 0.1482326877 1.718640e-02
##  [892,] 0.424154946 0.3063341278 7.374710e-02
##  [893,] 0.321175019 0.4163379880 1.798991e-01
##  [894,] 0.306334128 0.4241549461 1.957638e-01
##  [895,] 0.291090430 0.4308138364 2.125348e-01
##  [896,] 0.377630828 0.0906313987 7.250512e-03
##  [897,] 0.386693968 0.3625255950 1.132892e-01
##  [898,] 0.386693968 0.3625255950 1.132892e-01
##  [899,] 0.377630828 0.0906313987 7.250512e-03
##  [900,] 0.266544426 0.0339238361 1.439193e-03
##  [901,] 0.227598100 0.4443581954 2.891855e-01
##  [902,] 0.444093854 0.2114732637 3.356718e-02
##  [903,] 0.443086838 0.2436977611 4.467792e-02
##  [904,] 0.438655970 0.1794501695 2.447048e-02
##  [905,] 0.340371253 0.0654560102 4.195898e-03
##  [906,] 0.426168977 0.1482326877 1.718640e-02
##  [907,] 0.444358195 0.2275981001 3.885821e-02
##  [908,] 0.340371253 0.0654560102 4.195898e-03
##  [909,] 0.318229499 0.0540389715 3.058810e-03
##  [910,] 0.426168977 0.1482326877 1.718640e-02
##  [911,] 0.444093854 0.2114732637 3.356718e-02
##  [912,] 0.349346279 0.3975319727 1.507880e-01
##  [913,] 0.436239133 0.2755194522 5.800410e-02
##  [914,] 0.406028666 0.1184250277 1.151354e-02
##  [915,] 0.318229499 0.0540389715 3.058810e-03
##  [916,] 0.349346279 0.3975319727 1.507880e-01
##  [917,] 0.266544426 0.0339238361 1.439193e-03
##  [918,] 0.211473264 0.4440938538 3.108657e-01
##  [919,] 0.179450170 0.4386559699 3.574234e-01
##  [920,] 0.321175019 0.4163379880 1.798991e-01
##  [921,] 0.444358195 0.2275981001 3.885821e-02
##  [922,] 0.204487093 0.0179374643 5.244873e-04
##  [923,] 0.397531973 0.3493462791 1.023338e-01
##  [924,] 0.406028666 0.1184250277 1.151354e-02
##  [925,] 0.259696720 0.4403553087 2.488965e-01
##  [926,] 0.243697761 0.4430868383 2.685375e-01
##  [927,] 0.397531973 0.3493462791 1.023338e-01
##  [928,] 0.440355309 0.2596967205 5.105149e-02
##  [929,] 0.318229499 0.0540389715 3.058810e-03
##  [930,] 0.046838810 0.0007678494 4.195898e-06
##  [931,] 0.424154946 0.3063341278 7.374710e-02
##  [932,] 0.406028666 0.1184250277 1.151354e-02
##  [933,] 0.392899701 0.1042386963 9.218388e-03
##  [934,] 0.362525595 0.3866939680 1.374912e-01
##  [935,] 0.335537578 0.4074384881 1.649156e-01
##  [936,] 0.417093250 0.1331148669 1.416116e-02
##  [937,] 0.360146521 0.0776786613 5.584740e-03
##  [938,] 0.426168977 0.1482326877 1.718640e-02
##  [939,] 0.169380014 0.0116813803 2.685375e-04
##  [940,] 0.436239133 0.2755194522 5.800410e-02
##  [941,] 0.424154946 0.3063341278 7.374710e-02
##  [942,] 0.416337988 0.3211750193 8.258786e-02
##  [943,] 0.407438488 0.3355375785 9.210835e-02
##  [944,] 0.227598100 0.4443581954 2.891855e-01
##  [945,] 0.335537578 0.4074384881 1.649156e-01
##  [946,] 0.416337988 0.3211750193 8.258786e-02
##  [947,] 0.321175019 0.4163379880 1.798991e-01
##  [948,] 0.340371253 0.0654560102 4.195898e-03
##  [949,] 0.335537578 0.4074384881 1.649156e-01
##  [950,] 0.440355309 0.2596967205 5.105149e-02
##  [951,] 0.424154946 0.3063341278 7.374710e-02
##  [952,] 0.386693968 0.3625255950 1.132892e-01
##  [953,] 0.397531973 0.3493462791 1.023338e-01
##  [954,] 0.392899701 0.1042386963 9.218388e-03
##  [955,] 0.340371253 0.0654560102 4.195898e-03
##  [956,] 0.416337988 0.3211750193 8.258786e-02
##  [957,] 0.275519452 0.4362391326 2.302373e-01
##  [958,] 0.397531973 0.3493462791 1.023338e-01
##  [959,] 0.440355309 0.2596967205 5.105149e-02
##  [960,] 0.375000000 0.3750000000 1.250000e-01
##  [961,] 0.386693968 0.3625255950 1.132892e-01
##  [962,] 0.259696720 0.4403553087 2.488965e-01
##  [963,] 0.416337988 0.3211750193 8.258786e-02
##  [964,] 0.335537578 0.4074384881 1.649156e-01
##  [965,] 0.349346279 0.3975319727 1.507880e-01
##  [966,] 0.407438488 0.3355375785 9.210835e-02
##  [967,] 0.416337988 0.3211750193 8.258786e-02
##  [968,] 0.443086838 0.2436977611 4.467792e-02
##  [969,] 0.386693968 0.3625255950 1.132892e-01
##  [970,] 0.397531973 0.3493462791 1.023338e-01
##  [971,] 0.416337988 0.3211750193 8.258786e-02
##  [972,] 0.375000000 0.3750000000 1.250000e-01
##  [973,] 0.259696720 0.4403553087 2.488965e-01
##  [974,] 0.006684066 0.1314532913 8.617494e-01
##  [975,] 0.386693968 0.3625255950 1.132892e-01
##  [976,] 0.275519452 0.4362391326 2.302373e-01
##  [977,] 0.444358195 0.2275981001 3.885821e-02
##  [978,] 0.424154946 0.3063341278 7.374710e-02
##  [979,] 0.375000000 0.3750000000 1.250000e-01
##  [980,] 0.243697761 0.4430868383 2.685375e-01
##  [981,] 0.407438488 0.3355375785 9.210835e-02
##  [982,] 0.293645732 0.0435030714 2.148300e-03
##  [983,] 0.195398778 0.4422182874 3.336033e-01
##  [984,] 0.179450170 0.4386559699 3.574234e-01
##  [985,] 0.397531973 0.3493462791 1.023338e-01
##  [986,] 0.443086838 0.2436977611 4.467792e-02
##  [987,] 0.433331375 0.1637029640 2.061445e-02
##  [988,] 0.195398778 0.4422182874 3.336033e-01
##  [989,] 0.416337988 0.3211750193 8.258786e-02
##  [990,] 0.318229499 0.0540389715 3.058810e-03
##  [991,] 0.360146521 0.0776786613 5.584740e-03
##  [992,] 0.362525595 0.3866939680 1.374912e-01
##  [993,] 0.266544426 0.0339238361 1.439193e-03
##  [994,] 0.440355309 0.2596967205 5.105149e-02
##  [995,] 0.444093854 0.2114732637 3.356718e-02
##  [996,] 0.438655970 0.1794501695 2.447048e-02
##  [997,] 0.204487093 0.0179374643 5.244873e-04
##  [998,] 0.340371253 0.0654560102 4.195898e-03
##  [999,] 0.436239133 0.2755194522 5.800410e-02
## [1000,] 0.442218287 0.1953987782 2.877966e-02
## [1001,] 0.243697761 0.4430868383 2.685375e-01
## [1002,] 0.148232688 0.4261689772 4.084119e-01
## [1003,] 0.416337988 0.3211750193 8.258786e-02
## [1004,] 0.443086838 0.2436977611 4.467792e-02
## [1005,] 0.291090430 0.4308138364 2.125348e-01
## [1006,] 0.407438488 0.3355375785 9.210835e-02
## [1007,] 0.291090430 0.4308138364 2.125348e-01
## [1008,] 0.321175019 0.4163379880 1.798991e-01
## [1009,] 0.417093250 0.1331148669 1.416116e-02
## [1010,] 0.306334128 0.4241549461 1.957638e-01
## [1011,] 0.406028666 0.1184250277 1.151354e-02
## [1012,] 0.306334128 0.4241549461 1.957638e-01
## [1013,] 0.444093854 0.2114732637 3.356718e-02
## [1014,] 0.392899701 0.1042386963 9.218388e-03
## [1015,] 0.440355309 0.2596967205 5.105149e-02
## [1016,] 0.416337988 0.3211750193 8.258786e-02
## [1017,] 0.375000000 0.3750000000 1.250000e-01
## [1018,] 0.362525595 0.3866939680 1.374912e-01
## [1019,] 0.443086838 0.2436977611 4.467792e-02
## [1020,] 0.360146521 0.0776786613 5.584740e-03
## [1021,] 0.406028666 0.1184250277 1.151354e-02
## [1022,] 0.349346279 0.3975319727 1.507880e-01
## [1023,] 0.436239133 0.2755194522 5.800410e-02
## [1024,] 0.227598100 0.4443581954 2.891855e-01
## [1025,] 0.392899701 0.1042386963 9.218388e-03
## [1026,] 0.360146521 0.0776786613 5.584740e-03
## [1027,] 0.293645732 0.0435030714 2.148300e-03
## [1028,] 0.362525595 0.3866939680 1.374912e-01
## [1029,] 0.179450170 0.4386559699 3.574234e-01
## [1030,] 0.433331375 0.1637029640 2.061445e-02
## [1031,] 0.169380014 0.0116813803 2.685375e-04
## [1032,] 0.291090430 0.4308138364 2.125348e-01
## [1033,] 0.163702964 0.4333313752 3.823512e-01
## [1034,] 0.430813836 0.2910904300 6.556091e-02
## [1035,] 0.375000000 0.3750000000 1.250000e-01
## [1036,] 0.438655970 0.1794501695 2.447048e-02
## [1037,] 0.293645732 0.0435030714 2.148300e-03
## [1038,] 0.407438488 0.3355375785 9.210835e-02
## [1039,] 0.169380014 0.0116813803 2.685375e-04
## [1040,] 0.163702964 0.4333313752 3.823512e-01
## [1041,] 0.424154946 0.3063341278 7.374710e-02
## [1042,] 0.349346279 0.3975319727 1.507880e-01
## [1043,] 0.407438488 0.3355375785 9.210835e-02
## [1044,] 0.430813836 0.2910904300 6.556091e-02
## [1045,] 0.443086838 0.2436977611 4.467792e-02
## [1046,] 0.440355309 0.2596967205 5.105149e-02
## [1047,] 0.349346279 0.3975319727 1.507880e-01
## [1048,] 0.426168977 0.1482326877 1.718640e-02
## [1049,] 0.416337988 0.3211750193 8.258786e-02
## [1050,] 0.433331375 0.1637029640 2.061445e-02
## [1051,] 0.417093250 0.1331148669 1.416116e-02
## [1052,] 0.407438488 0.3355375785 9.210835e-02
## [1053,] 0.424154946 0.3063341278 7.374710e-02
## [1054,] 0.362525595 0.3866939680 1.374912e-01
## [1055,] 0.291090430 0.4308138364 2.125348e-01
## [1056,] 0.375000000 0.3750000000 1.250000e-01
## [1057,] 0.397531973 0.3493462791 1.023338e-01
## [1058,] 0.443086838 0.2436977611 4.467792e-02
## [1059,] 0.131453291 0.0066840657 1.132892e-04
## [1060,] 0.211473264 0.4440938538 3.108657e-01
## [1061,] 0.275519452 0.4362391326 2.302373e-01
## [1062,] 0.195398778 0.4422182874 3.336033e-01
## [1063,] 0.424154946 0.3063341278 7.374710e-02
## [1064,] 0.430813836 0.2910904300 6.556091e-02
## [1065,] 0.360146521 0.0776786613 5.584740e-03
## [1066,] 0.444093854 0.2114732637 3.356718e-02
## [1067,] 0.293645732 0.0435030714 2.148300e-03
## [1068,] 0.340371253 0.0654560102 4.195898e-03
## [1069,] 0.416337988 0.3211750193 8.258786e-02
## [1070,] 0.444358195 0.2275981001 3.885821e-02
## [1071,] 0.417093250 0.1331148669 1.416116e-02
## [1072,] 0.424154946 0.3063341278 7.374710e-02
## [1073,] 0.386693968 0.3625255950 1.132892e-01
## [1074,] 0.416337988 0.3211750193 8.258786e-02
## [1075,] 0.275519452 0.4362391326 2.302373e-01
## [1076,] 0.443086838 0.2436977611 4.467792e-02
## [1077,] 0.054038972 0.3182294988 6.246727e-01
## [1078,] 0.377630828 0.0906313987 7.250512e-03
## [1079,] 0.416337988 0.3211750193 8.258786e-02
## [1080,] 0.440355309 0.2596967205 5.105149e-02
## [1081,] 0.443086838 0.2436977611 4.467792e-02
## [1082,] 0.227598100 0.4443581954 2.891855e-01
## [1083,] 0.444093854 0.2114732637 3.356718e-02
## [1084,] 0.293645732 0.0435030714 2.148300e-03
## [1085,] 0.321175019 0.4163379880 1.798991e-01
## [1086,] 0.407438488 0.3355375785 9.210835e-02
## [1087,] 0.436239133 0.2755194522 5.800410e-02
## [1088,] 0.377630828 0.0906313987 7.250512e-03
## [1089,] 0.426168977 0.1482326877 1.718640e-02
## [1090,] 0.335537578 0.4074384881 1.649156e-01
## [1091,] 0.335537578 0.4074384881 1.649156e-01
## [1092,] 0.306334128 0.4241549461 1.957638e-01
## [1093,] 0.397531973 0.3493462791 1.023338e-01
## [1094,] 0.131453291 0.0066840657 1.132892e-04
## [1095,] 0.043503071 0.2936457319 6.607029e-01
## [1096,] 0.444093854 0.2114732637 3.356718e-02
## [1097,] 0.321175019 0.4163379880 1.798991e-01
## [1098,] 0.433331375 0.1637029640 2.061445e-02
## [1099,] 0.211473264 0.4440938538 3.108657e-01
## [1100,] 0.444358195 0.2275981001 3.885821e-02
## [1101,] 0.195398778 0.4422182874 3.336033e-01
## [1102,] 0.148232688 0.4261689772 4.084119e-01
## [1103,] 0.407438488 0.3355375785 9.210835e-02
## [1104,] 0.266544426 0.0339238361 1.439193e-03
## [1105,] 0.000000000 0.0000000000 1.000000e+00
## [1106,] 0.349346279 0.3975319727 1.507880e-01
## [1107,] 0.243697761 0.4430868383 2.685375e-01
## [1108,] 0.335537578 0.4074384881 1.649156e-01
## [1109,] 0.416337988 0.3211750193 8.258786e-02
## [1110,] 0.392899701 0.1042386963 9.218388e-03
## [1111,] 0.375000000 0.3750000000 1.250000e-01
## [1112,] 0.397531973 0.3493462791 1.023338e-01
## [1113,] 0.444358195 0.2275981001 3.885821e-02
## [1114,] 0.321175019 0.4163379880 1.798991e-01
## [1115,] 0.442218287 0.1953987782 2.877966e-02
## [1116,] 0.335537578 0.4074384881 1.649156e-01
## [1117,] 0.444358195 0.2275981001 3.885821e-02
## [1118,] 0.163702964 0.4333313752 3.823512e-01
## [1119,] 0.204487093 0.0179374643 5.244873e-04
## [1120,] 0.179450170 0.4386559699 3.574234e-01
## [1121,] 0.430813836 0.2910904300 6.556091e-02
## [1122,] 0.426168977 0.1482326877 1.718640e-02
## [1123,] 0.444093854 0.2114732637 3.356718e-02
## [1124,] 0.266544426 0.0339238361 1.439193e-03
## [1125,] 0.377630828 0.0906313987 7.250512e-03
## [1126,] 0.417093250 0.1331148669 1.416116e-02
## [1127,] 0.360146521 0.0776786613 5.584740e-03
## [1128,] 0.406028666 0.1184250277 1.151354e-02
## [1129,] 0.306334128 0.4241549461 1.957638e-01
## [1130,] 0.236850055 0.0253767916 9.063140e-04
## [1131,] 0.377630828 0.0906313987 7.250512e-03
## [1132,] 0.397531973 0.3493462791 1.023338e-01
## [1133,] 0.424154946 0.3063341278 7.374710e-02
## [1134,] 0.440355309 0.2596967205 5.105149e-02
## [1135,] 0.306334128 0.4241549461 1.957638e-01
## [1136,] 0.266544426 0.0339238361 1.439193e-03
## [1137,] 0.375000000 0.3750000000 1.250000e-01
## [1138,] 0.433331375 0.1637029640 2.061445e-02
## [1139,] 0.118425028 0.4060286664 4.640328e-01
## [1140,] 0.259696720 0.4403553087 2.488965e-01
## [1141,] 0.397531973 0.3493462791 1.023338e-01
## [1142,] 0.275519452 0.4362391326 2.302373e-01
## [1143,] 0.426168977 0.1482326877 1.718640e-02
## [1144,] 0.204487093 0.0179374643 5.244873e-04
## [1145,] 0.430813836 0.2910904300 6.556091e-02
## [1146,] 0.438655970 0.1794501695 2.447048e-02
## [1147,] 0.169380014 0.0116813803 2.685375e-04
## [1148,] 0.362525595 0.3866939680 1.374912e-01
## [1149,] 0.243697761 0.4430868383 2.685375e-01
## [1150,] 0.424154946 0.3063341278 7.374710e-02
## [1151,] 0.362525595 0.3866939680 1.374912e-01
## [1152,] 0.291090430 0.4308138364 2.125348e-01
## [1153,] 0.406028666 0.1184250277 1.151354e-02
## [1154,] 0.362525595 0.3866939680 1.374912e-01
## [1155,] 0.236850055 0.0253767916 9.063140e-04
## [1156,] 0.321175019 0.4163379880 1.798991e-01
## [1157,] 0.266544426 0.0339238361 1.439193e-03
## [1158,] 0.259696720 0.4403553087 2.488965e-01
## [1159,] 0.430813836 0.2910904300 6.556091e-02
## [1160,] 0.443086838 0.2436977611 4.467792e-02
## [1161,] 0.444358195 0.2275981001 3.885821e-02
## [1162,] 0.406028666 0.1184250277 1.151354e-02
## [1163,] 0.386693968 0.3625255950 1.132892e-01
## [1164,] 0.433331375 0.1637029640 2.061445e-02
## [1165,] 0.335537578 0.4074384881 1.649156e-01
## [1166,] 0.362525595 0.3866939680 1.374912e-01
## [1167,] 0.433331375 0.1637029640 2.061445e-02
## [1168,] 0.318229499 0.0540389715 3.058810e-03
## [1169,] 0.259696720 0.4403553087 2.488965e-01
## [1170,] 0.386693968 0.3625255950 1.132892e-01
## [1171,] 0.440355309 0.2596967205 5.105149e-02
## [1172,] 0.227598100 0.4443581954 2.891855e-01
## [1173,] 0.291090430 0.4308138364 2.125348e-01
## [1174,] 0.426168977 0.1482326877 1.718640e-02
## [1175,] 0.430813836 0.2910904300 6.556091e-02
## [1176,] 0.430813836 0.2910904300 6.556091e-02
## [1177,] 0.417093250 0.1331148669 1.416116e-02
## [1178,] 0.131453291 0.0066840657 1.132892e-04
## [1179,] 0.306334128 0.4241549461 1.957638e-01
## [1180,] 0.306334128 0.4241549461 1.957638e-01
## [1181,] 0.433331375 0.1637029640 2.061445e-02
## [1182,] 0.204487093 0.0179374643 5.244873e-04
## [1183,] 0.195398778 0.4422182874 3.336033e-01
## [1184,] 0.349346279 0.3975319727 1.507880e-01
## [1185,] 0.090631399 0.0030210466 3.356718e-05
## [1186,] 0.349346279 0.3975319727 1.507880e-01
## [1187,] 0.133114867 0.4170932496 4.356307e-01
## [1188,] 0.442218287 0.1953987782 2.877966e-02
## [1189,] 0.236850055 0.0253767916 9.063140e-04
## [1190,] 0.438655970 0.1794501695 2.447048e-02
## [1191,] 0.417093250 0.1331148669 1.416116e-02
## [1192,] 0.438655970 0.1794501695 2.447048e-02
## [1193,] 0.406028666 0.1184250277 1.151354e-02
## [1194,] 0.416337988 0.3211750193 8.258786e-02
## [1195,] 0.417093250 0.1331148669 1.416116e-02
## [1196,] 0.397531973 0.3493462791 1.023338e-01
## [1197,] 0.442218287 0.1953987782 2.877966e-02
## [1198,] 0.259696720 0.4403553087 2.488965e-01
## [1199,] 0.397531973 0.3493462791 1.023338e-01
## [1200,] 0.360146521 0.0776786613 5.584740e-03
## [1201,] 0.442218287 0.1953987782 2.877966e-02
## [1202,] 0.259696720 0.4403553087 2.488965e-01
## [1203,] 0.444358195 0.2275981001 3.885821e-02
## [1204,] 0.227598100 0.4443581954 2.891855e-01
## [1205,] 0.392899701 0.1042386963 9.218388e-03
## [1206,] 0.293645732 0.0435030714 2.148300e-03
## [1207,] 0.444093854 0.2114732637 3.356718e-02
## [1208,] 0.349346279 0.3975319727 1.507880e-01
## [1209,] 0.406028666 0.1184250277 1.151354e-02
## [1210,] 0.375000000 0.3750000000 1.250000e-01
## [1211,] 0.443086838 0.2436977611 4.467792e-02
## [1212,] 0.211473264 0.4440938538 3.108657e-01
## [1213,] 0.377630828 0.0906313987 7.250512e-03
## [1214,] 0.440355309 0.2596967205 5.105149e-02
## [1215,] 0.406028666 0.1184250277 1.151354e-02
## [1216,] 0.440355309 0.2596967205 5.105149e-02
## [1217,] 0.321175019 0.4163379880 1.798991e-01
## [1218,] 0.433331375 0.1637029640 2.061445e-02
## [1219,] 0.430813836 0.2910904300 6.556091e-02
## [1220,] 0.362525595 0.3866939680 1.374912e-01
## [1221,] 0.046838810 0.0007678494 4.195898e-06
## [1222,] 0.321175019 0.4163379880 1.798991e-01
## [1223,] 0.169380014 0.0116813803 2.685375e-04
## [1224,] 0.375000000 0.3750000000 1.250000e-01
## [1225,] 0.417093250 0.1331148669 1.416116e-02
## [1226,] 0.392899701 0.1042386963 9.218388e-03
## [1227,] 0.430813836 0.2910904300 6.556091e-02
## [1228,] 0.443086838 0.2436977611 4.467792e-02
## [1229,] 0.386693968 0.3625255950 1.132892e-01
## [1230,] 0.407438488 0.3355375785 9.210835e-02
## [1231,] 0.243697761 0.4430868383 2.685375e-01
## [1232,] 0.362525595 0.3866939680 1.374912e-01
## [1233,] 0.444093854 0.2114732637 3.356718e-02
## [1234,] 0.417093250 0.1331148669 1.416116e-02
## [1235,] 0.335537578 0.4074384881 1.649156e-01
## [1236,] 0.321175019 0.4163379880 1.798991e-01
## [1237,] 0.442218287 0.1953987782 2.877966e-02
## [1238,] 0.306334128 0.4241549461 1.957638e-01
## [1239,] 0.306334128 0.4241549461 1.957638e-01
## [1240,] 0.266544426 0.0339238361 1.439193e-03
## [1241,] 0.433331375 0.1637029640 2.061445e-02
## [1242,] 0.360146521 0.0776786613 5.584740e-03
## [1243,] 0.430813836 0.2910904300 6.556091e-02
## [1244,] 0.291090430 0.4308138364 2.125348e-01
## [1245,] 0.386693968 0.3625255950 1.132892e-01
## [1246,] 0.436239133 0.2755194522 5.800410e-02
## [1247,] 0.430813836 0.2910904300 6.556091e-02
## [1248,] 0.406028666 0.1184250277 1.151354e-02
## [1249,] 0.090631399 0.0030210466 3.356718e-05
## [1250,] 0.430813836 0.2910904300 6.556091e-02
## [1251,] 0.243697761 0.4430868383 2.685375e-01
## [1252,] 0.444093854 0.2114732637 3.356718e-02
## [1253,] 0.204487093 0.0179374643 5.244873e-04
## [1254,] 0.306334128 0.4241549461 1.957638e-01
## [1255,] 0.118425028 0.4060286664 4.640328e-01
## [1256,] 0.397531973 0.3493462791 1.023338e-01
## [1257,] 0.444358195 0.2275981001 3.885821e-02
## [1258,] 0.433331375 0.1637029640 2.061445e-02
## [1259,] 0.443086838 0.2436977611 4.467792e-02
## [1260,] 0.443086838 0.2436977611 4.467792e-02
## [1261,] 0.433331375 0.1637029640 2.061445e-02
## [1262,] 0.293645732 0.0435030714 2.148300e-03
## [1263,] 0.204487093 0.0179374643 5.244873e-04
## [1264,] 0.195398778 0.4422182874 3.336033e-01
## [1265,] 0.236850055 0.0253767916 9.063140e-04
## [1266,] 0.362525595 0.3866939680 1.374912e-01
## [1267,] 0.169380014 0.0116813803 2.685375e-04
## [1268,] 0.179450170 0.4386559699 3.574234e-01
## [1269,] 0.440355309 0.2596967205 5.105149e-02
## [1270,] 0.306334128 0.4241549461 1.957638e-01
## [1271,] 0.360146521 0.0776786613 5.584740e-03
## [1272,] 0.444358195 0.2275981001 3.885821e-02
## [1273,] 0.054038972 0.3182294988 6.246727e-01
## [1274,] 0.169380014 0.0116813803 2.685375e-04
## [1275,] 0.386693968 0.3625255950 1.132892e-01
## [1276,] 0.433331375 0.1637029640 2.061445e-02
## [1277,] 0.407438488 0.3355375785 9.210835e-02
## [1278,] 0.291090430 0.4308138364 2.125348e-01
## [1279,] 0.438655970 0.1794501695 2.447048e-02
## [1280,] 0.131453291 0.0066840657 1.132892e-04
## [1281,] 0.440355309 0.2596967205 5.105149e-02
## [1282,] 0.406028666 0.1184250277 1.151354e-02
## [1283,] 0.438655970 0.1794501695 2.447048e-02
## [1284,] 0.340371253 0.0654560102 4.195898e-03
## [1285,] 0.440355309 0.2596967205 5.105149e-02
## [1286,] 0.291090430 0.4308138364 2.125348e-01
## [1287,] 0.424154946 0.3063341278 7.374710e-02
## [1288,] 0.440355309 0.2596967205 5.105149e-02
## [1289,] 0.259696720 0.4403553087 2.488965e-01
## [1290,] 0.291090430 0.4308138364 2.125348e-01
## [1291,] 0.438655970 0.1794501695 2.447048e-02
## [1292,] 0.430813836 0.2910904300 6.556091e-02
## [1293,] 0.318229499 0.0540389715 3.058810e-03
## [1294,] 0.406028666 0.1184250277 1.151354e-02
## [1295,] 0.444093854 0.2114732637 3.356718e-02
## [1296,] 0.340371253 0.0654560102 4.195898e-03
## [1297,] 0.436239133 0.2755194522 5.800410e-02
## [1298,] 0.349346279 0.3975319727 1.507880e-01
## [1299,] 0.291090430 0.4308138364 2.125348e-01
## [1300,] 0.444358195 0.2275981001 3.885821e-02
## [1301,] 0.436239133 0.2755194522 5.800410e-02
## [1302,] 0.204487093 0.0179374643 5.244873e-04
## [1303,] 0.443086838 0.2436977611 4.467792e-02
## [1304,] 0.443086838 0.2436977611 4.467792e-02
## [1305,] 0.349346279 0.3975319727 1.507880e-01
## [1306,] 0.011681380 0.1693800141 8.186701e-01
## [1307,] 0.318229499 0.0540389715 3.058810e-03
## [1308,] 0.266544426 0.0339238361 1.439193e-03
## [1309,] 0.318229499 0.0540389715 3.058810e-03
## [1310,] 0.417093250 0.1331148669 1.416116e-02
## [1311,] 0.349346279 0.3975319727 1.507880e-01
## [1312,] 0.169380014 0.0116813803 2.685375e-04
## [1313,] 0.397531973 0.3493462791 1.023338e-01
## [1314,] 0.426168977 0.1482326877 1.718640e-02
## [1315,] 0.397531973 0.3493462791 1.023338e-01
## [1316,] 0.392899701 0.1042386963 9.218388e-03
## [1317,] 0.397531973 0.3493462791 1.023338e-01
## [1318,] 0.375000000 0.3750000000 1.250000e-01
## [1319,] 0.443086838 0.2436977611 4.467792e-02
## [1320,] 0.349346279 0.3975319727 1.507880e-01
## [1321,] 0.392899701 0.1042386963 9.218388e-03
## [1322,] 0.386693968 0.3625255950 1.132892e-01
## [1323,] 0.275519452 0.4362391326 2.302373e-01
## [1324,] 0.407438488 0.3355375785 9.210835e-02
## [1325,] 0.321175019 0.4163379880 1.798991e-01
## [1326,] 0.406028666 0.1184250277 1.151354e-02
## [1327,] 0.291090430 0.4308138364 2.125348e-01
## [1328,] 0.433331375 0.1637029640 2.061445e-02
## [1329,] 0.417093250 0.1331148669 1.416116e-02
## [1330,] 0.417093250 0.1331148669 1.416116e-02
## [1331,] 0.440355309 0.2596967205 5.105149e-02
## [1332,] 0.436239133 0.2755194522 5.800410e-02
## [1333,] 0.243697761 0.4430868383 2.685375e-01
## [1334,] 0.416337988 0.3211750193 8.258786e-02
## [1335,] 0.397531973 0.3493462791 1.023338e-01
## [1336,] 0.426168977 0.1482326877 1.718640e-02
## [1337,] 0.430813836 0.2910904300 6.556091e-02
## [1338,] 0.243697761 0.4430868383 2.685375e-01
## [1339,] 0.424154946 0.3063341278 7.374710e-02
## [1340,] 0.438655970 0.1794501695 2.447048e-02
## [1341,] 0.397531973 0.3493462791 1.023338e-01
## [1342,] 0.275519452 0.4362391326 2.302373e-01
## [1343,] 0.444093854 0.2114732637 3.356718e-02
## [1344,] 0.424154946 0.3063341278 7.374710e-02
## [1345,] 0.275519452 0.4362391326 2.302373e-01
## [1346,] 0.349346279 0.3975319727 1.507880e-01
## [1347,] 0.440355309 0.2596967205 5.105149e-02
## [1348,] 0.335537578 0.4074384881 1.649156e-01
## [1349,] 0.318229499 0.0540389715 3.058810e-03
## [1350,] 0.335537578 0.4074384881 1.649156e-01
## [1351,] 0.349346279 0.3975319727 1.507880e-01
## [1352,] 0.349346279 0.3975319727 1.507880e-01
## [1353,] 0.340371253 0.0654560102 4.195898e-03
## [1354,] 0.375000000 0.3750000000 1.250000e-01
## [1355,] 0.195398778 0.4422182874 3.336033e-01
## [1356,] 0.204487093 0.0179374643 5.244873e-04
## [1357,] 0.321175019 0.4163379880 1.798991e-01
## [1358,] 0.291090430 0.4308138364 2.125348e-01
## [1359,] 0.386693968 0.3625255950 1.132892e-01
## [1360,] 0.362525595 0.3866939680 1.374912e-01
## [1361,] 0.375000000 0.3750000000 1.250000e-01
## [1362,] 0.375000000 0.3750000000 1.250000e-01
## [1363,] 0.430813836 0.2910904300 6.556091e-02
## [1364,] 0.407438488 0.3355375785 9.210835e-02
## [1365,] 0.386693968 0.3625255950 1.132892e-01
## [1366,] 0.046838810 0.0007678494 4.195898e-06
## [1367,] 0.275519452 0.4362391326 2.302373e-01
## [1368,] 0.424154946 0.3063341278 7.374710e-02
## [1369,] 0.436239133 0.2755194522 5.800410e-02
## [1370,] 0.406028666 0.1184250277 1.151354e-02
## [1371,] 0.406028666 0.1184250277 1.151354e-02
## [1372,] 0.430813836 0.2910904300 6.556091e-02
## [1373,] 0.259696720 0.4403553087 2.488965e-01
## [1374,] 0.104238696 0.3928997013 4.936432e-01
## [1375,] 0.392899701 0.1042386963 9.218388e-03
## [1376,] 0.375000000 0.3750000000 1.250000e-01
## [1377,] 0.440355309 0.2596967205 5.105149e-02
## [1378,] 0.433331375 0.1637029640 2.061445e-02
## [1379,] 0.417093250 0.1331148669 1.416116e-02
## [1380,] 0.321175019 0.4163379880 1.798991e-01
## [1381,] 0.430813836 0.2910904300 6.556091e-02
## [1382,] 0.438655970 0.1794501695 2.447048e-02
## [1383,] 0.444093854 0.2114732637 3.356718e-02
## [1384,] 0.243697761 0.4430868383 2.685375e-01
## [1385,] 0.416337988 0.3211750193 8.258786e-02
## [1386,] 0.426168977 0.1482326877 1.718640e-02
## [1387,] 0.131453291 0.0066840657 1.132892e-04
## [1388,] 0.444358195 0.2275981001 3.885821e-02
## [1389,] 0.340371253 0.0654560102 4.195898e-03
## [1390,] 0.306334128 0.4241549461 1.957638e-01
## [1391,] 0.236850055 0.0253767916 9.063140e-04
## [1392,] 0.392899701 0.1042386963 9.218388e-03
## [1393,] 0.424154946 0.3063341278 7.374710e-02
## [1394,] 0.377630828 0.0906313987 7.250512e-03
## [1395,] 0.440355309 0.2596967205 5.105149e-02
## [1396,] 0.293645732 0.0435030714 2.148300e-03
## [1397,] 0.406028666 0.1184250277 1.151354e-02
## [1398,] 0.436239133 0.2755194522 5.800410e-02
## [1399,] 0.424154946 0.3063341278 7.374710e-02
## [1400,] 0.377630828 0.0906313987 7.250512e-03
## [1401,] 0.243697761 0.4430868383 2.685375e-01
## [1402,] 0.417093250 0.1331148669 1.416116e-02
## [1403,] 0.340371253 0.0654560102 4.195898e-03
## [1404,] 0.430813836 0.2910904300 6.556091e-02
## [1405,] 0.375000000 0.3750000000 1.250000e-01
## [1406,] 0.438655970 0.1794501695 2.447048e-02
## [1407,] 0.397531973 0.3493462791 1.023338e-01
## [1408,] 0.426168977 0.1482326877 1.718640e-02
## [1409,] 0.179450170 0.4386559699 3.574234e-01
## [1410,] 0.424154946 0.3063341278 7.374710e-02
## [1411,] 0.386693968 0.3625255950 1.132892e-01
## [1412,] 0.275519452 0.4362391326 2.302373e-01
## [1413,] 0.362525595 0.3866939680 1.374912e-01
## [1414,] 0.377630828 0.0906313987 7.250512e-03
## [1415,] 0.426168977 0.1482326877 1.718640e-02
## [1416,] 0.349346279 0.3975319727 1.507880e-01
## [1417,] 0.321175019 0.4163379880 1.798991e-01
## [1418,] 0.443086838 0.2436977611 4.467792e-02
## [1419,] 0.426168977 0.1482326877 1.718640e-02
## [1420,] 0.438655970 0.1794501695 2.447048e-02
## [1421,] 0.306334128 0.4241549461 1.957638e-01
## [1422,] 0.179450170 0.4386559699 3.574234e-01
## [1423,] 0.417093250 0.1331148669 1.416116e-02
## [1424,] 0.424154946 0.3063341278 7.374710e-02
## [1425,] 0.000000000 0.0000000000 1.000000e+00
## [1426,] 0.349346279 0.3975319727 1.507880e-01
## [1427,] 0.211473264 0.4440938538 3.108657e-01
## [1428,] 0.417093250 0.1331148669 1.416116e-02
## [1429,] 0.340371253 0.0654560102 4.195898e-03
## [1430,] 0.275519452 0.4362391326 2.302373e-01
## [1431,] 0.275519452 0.4362391326 2.302373e-01
## [1432,] 0.426168977 0.1482326877 1.718640e-02
## [1433,] 0.416337988 0.3211750193 8.258786e-02
## [1434,] 0.275519452 0.4362391326 2.302373e-01
## [1435,] 0.340371253 0.0654560102 4.195898e-03
## [1436,] 0.442218287 0.1953987782 2.877966e-02
## [1437,] 0.275519452 0.4362391326 2.302373e-01
## [1438,] 0.169380014 0.0116813803 2.685375e-04
## [1439,] 0.211473264 0.4440938538 3.108657e-01
## [1440,] 0.377630828 0.0906313987 7.250512e-03
## [1441,] 0.362525595 0.3866939680 1.374912e-01
## [1442,] 0.444093854 0.2114732637 3.356718e-02
## [1443,] 0.291090430 0.4308138364 2.125348e-01
## [1444,] 0.444358195 0.2275981001 3.885821e-02
## [1445,] 0.436239133 0.2755194522 5.800410e-02
## [1446,] 0.054038972 0.3182294988 6.246727e-01
## [1447,] 0.375000000 0.3750000000 1.250000e-01
## [1448,] 0.416337988 0.3211750193 8.258786e-02
## [1449,] 0.440355309 0.2596967205 5.105149e-02
## [1450,] 0.417093250 0.1331148669 1.416116e-02
## [1451,] 0.397531973 0.3493462791 1.023338e-01
## [1452,] 0.204487093 0.0179374643 5.244873e-04
## [1453,] 0.406028666 0.1184250277 1.151354e-02
## [1454,] 0.377630828 0.0906313987 7.250512e-03
## [1455,] 0.306334128 0.4241549461 1.957638e-01
## [1456,] 0.335537578 0.4074384881 1.649156e-01
## [1457,] 0.377630828 0.0906313987 7.250512e-03
## [1458,] 0.406028666 0.1184250277 1.151354e-02
## [1459,] 0.321175019 0.4163379880 1.798991e-01
## [1460,] 0.392899701 0.1042386963 9.218388e-03
## [1461,] 0.362525595 0.3866939680 1.374912e-01
## [1462,] 0.440355309 0.2596967205 5.105149e-02
## [1463,] 0.397531973 0.3493462791 1.023338e-01
## [1464,] 0.442218287 0.1953987782 2.877966e-02
## [1465,] 0.236850055 0.0253767916 9.063140e-04
## [1466,] 0.321175019 0.4163379880 1.798991e-01
## [1467,] 0.444358195 0.2275981001 3.885821e-02
## [1468,] 0.397531973 0.3493462791 1.023338e-01
## [1469,] 0.438655970 0.1794501695 2.447048e-02
## [1470,] 0.211473264 0.4440938538 3.108657e-01
## [1471,] 0.430813836 0.2910904300 6.556091e-02
## [1472,] 0.090631399 0.0030210466 3.356718e-05
## [1473,] 0.318229499 0.0540389715 3.058810e-03
## [1474,] 0.362525595 0.3866939680 1.374912e-01
## [1475,] 0.275519452 0.4362391326 2.302373e-01
## [1476,] 0.046838810 0.0007678494 4.195898e-06
## [1477,] 0.433331375 0.1637029640 2.061445e-02
## [1478,] 0.416337988 0.3211750193 8.258786e-02
## [1479,] 0.306334128 0.4241549461 1.957638e-01
## [1480,] 0.436239133 0.2755194522 5.800410e-02
## [1481,] 0.349346279 0.3975319727 1.507880e-01
## [1482,] 0.386693968 0.3625255950 1.132892e-01
## [1483,] 0.362525595 0.3866939680 1.374912e-01
## [1484,] 0.442218287 0.1953987782 2.877966e-02
## [1485,] 0.444093854 0.2114732637 3.356718e-02
## [1486,] 0.440355309 0.2596967205 5.105149e-02
## [1487,] 0.349346279 0.3975319727 1.507880e-01
## [1488,] 0.349346279 0.3975319727 1.507880e-01
## [1489,] 0.430813836 0.2910904300 6.556091e-02
## [1490,] 0.426168977 0.1482326877 1.718640e-02
## [1491,] 0.430813836 0.2910904300 6.556091e-02
## [1492,] 0.227598100 0.4443581954 2.891855e-01
## [1493,] 0.195398778 0.4422182874 3.336033e-01
## [1494,] 0.375000000 0.3750000000 1.250000e-01
## [1495,] 0.306334128 0.4241549461 1.957638e-01
## [1496,] 0.440355309 0.2596967205 5.105149e-02
## [1497,] 0.360146521 0.0776786613 5.584740e-03
## [1498,] 0.118425028 0.4060286664 4.640328e-01
## [1499,] 0.426168977 0.1482326877 1.718640e-02
## [1500,] 0.440355309 0.2596967205 5.105149e-02
## [1501,] 0.293645732 0.0435030714 2.148300e-03
## [1502,] 0.306334128 0.4241549461 1.957638e-01
## [1503,] 0.424154946 0.3063341278 7.374710e-02
## [1504,] 0.321175019 0.4163379880 1.798991e-01
## [1505,] 0.306334128 0.4241549461 1.957638e-01
## [1506,] 0.179450170 0.4386559699 3.574234e-01
## [1507,] 0.443086838 0.2436977611 4.467792e-02
## [1508,] 0.444358195 0.2275981001 3.885821e-02
## [1509,] 0.291090430 0.4308138364 2.125348e-01
## [1510,] 0.259696720 0.4403553087 2.488965e-01
## [1511,] 0.416337988 0.3211750193 8.258786e-02
## [1512,] 0.340371253 0.0654560102 4.195898e-03
## [1513,] 0.243697761 0.4430868383 2.685375e-01
## [1514,] 0.335537578 0.4074384881 1.649156e-01
## [1515,] 0.392899701 0.1042386963 9.218388e-03
## [1516,] 0.163702964 0.4333313752 3.823512e-01
## [1517,] 0.436239133 0.2755194522 5.800410e-02
## [1518,] 0.377630828 0.0906313987 7.250512e-03
## [1519,] 0.335537578 0.4074384881 1.649156e-01
## [1520,] 0.436239133 0.2755194522 5.800410e-02
## [1521,] 0.259696720 0.4403553087 2.488965e-01
## [1522,] 0.407438488 0.3355375785 9.210835e-02
## [1523,] 0.131453291 0.0066840657 1.132892e-04
## [1524,] 0.426168977 0.1482326877 1.718640e-02
## [1525,] 0.444358195 0.2275981001 3.885821e-02
## [1526,] 0.436239133 0.2755194522 5.800410e-02
## [1527,] 0.000000000 0.0000000000 1.000000e+00
## [1528,] 0.392899701 0.1042386963 9.218388e-03
## [1529,] 0.440355309 0.2596967205 5.105149e-02
## [1530,] 0.442218287 0.1953987782 2.877966e-02
## [1531,] 0.430813836 0.2910904300 6.556091e-02
## [1532,] 0.306334128 0.4241549461 1.957638e-01
## [1533,] 0.416337988 0.3211750193 8.258786e-02
## [1534,] 0.227598100 0.4443581954 2.891855e-01
## [1535,] 0.360146521 0.0776786613 5.584740e-03
## [1536,] 0.360146521 0.0776786613 5.584740e-03
## [1537,] 0.416337988 0.3211750193 8.258786e-02
## [1538,] 0.163702964 0.4333313752 3.823512e-01
## [1539,] 0.275519452 0.4362391326 2.302373e-01
## [1540,] 0.444358195 0.2275981001 3.885821e-02
## [1541,] 0.436239133 0.2755194522 5.800410e-02
## [1542,] 0.397531973 0.3493462791 1.023338e-01
## [1543,] 0.430813836 0.2910904300 6.556091e-02
## [1544,] 0.436239133 0.2755194522 5.800410e-02
## [1545,] 0.362525595 0.3866939680 1.374912e-01
## [1546,] 0.444358195 0.2275981001 3.885821e-02
## [1547,] 0.362525595 0.3866939680 1.374912e-01
## [1548,] 0.211473264 0.4440938538 3.108657e-01
## [1549,] 0.259696720 0.4403553087 2.488965e-01
## [1550,] 0.375000000 0.3750000000 1.250000e-01
## [1551,] 0.417093250 0.1331148669 1.416116e-02
## [1552,] 0.227598100 0.4443581954 2.891855e-01
## [1553,] 0.440355309 0.2596967205 5.105149e-02
## [1554,] 0.417093250 0.1331148669 1.416116e-02
## [1555,] 0.340371253 0.0654560102 4.195898e-03
## [1556,] 0.375000000 0.3750000000 1.250000e-01
## [1557,] 0.349346279 0.3975319727 1.507880e-01
## [1558,] 0.169380014 0.0116813803 2.685375e-04
## [1559,] 0.397531973 0.3493462791 1.023338e-01
## [1560,] 0.227598100 0.4443581954 2.891855e-01
## [1561,] 0.440355309 0.2596967205 5.105149e-02
## [1562,] 0.406028666 0.1184250277 1.151354e-02
## [1563,] 0.444358195 0.2275981001 3.885821e-02
## [1564,] 0.148232688 0.4261689772 4.084119e-01
## [1565,] 0.438655970 0.1794501695 2.447048e-02
## [1566,] 0.195398778 0.4422182874 3.336033e-01
## [1567,] 0.426168977 0.1482326877 1.718640e-02
## [1568,] 0.335537578 0.4074384881 1.649156e-01
## [1569,] 0.417093250 0.1331148669 1.416116e-02
## [1570,] 0.426168977 0.1482326877 1.718640e-02
## [1571,] 0.444358195 0.2275981001 3.885821e-02
## [1572,] 0.227598100 0.4443581954 2.891855e-01
## [1573,] 0.375000000 0.3750000000 1.250000e-01
## [1574,] 0.443086838 0.2436977611 4.467792e-02
## [1575,] 0.375000000 0.3750000000 1.250000e-01
## [1576,] 0.227598100 0.4443581954 2.891855e-01
## [1577,] 0.444358195 0.2275981001 3.885821e-02
## [1578,] 0.163702964 0.4333313752 3.823512e-01
## [1579,] 0.266544426 0.0339238361 1.439193e-03
## [1580,] 0.321175019 0.4163379880 1.798991e-01
## [1581,] 0.204487093 0.0179374643 5.244873e-04
## [1582,] 0.438655970 0.1794501695 2.447048e-02
## [1583,] 0.046838810 0.0007678494 4.195898e-06
## [1584,] 0.430813836 0.2910904300 6.556091e-02
## [1585,] 0.443086838 0.2436977611 4.467792e-02
## [1586,] 0.444093854 0.2114732637 3.356718e-02
## [1587,] 0.163702964 0.4333313752 3.823512e-01
## [1588,] 0.416337988 0.3211750193 8.258786e-02
## [1589,] 0.406028666 0.1184250277 1.151354e-02
## [1590,] 0.442218287 0.1953987782 2.877966e-02
## [1591,] 0.442218287 0.1953987782 2.877966e-02
## [1592,] 0.416337988 0.3211750193 8.258786e-02
## [1593,] 0.424154946 0.3063341278 7.374710e-02
## [1594,] 0.444358195 0.2275981001 3.885821e-02
## [1595,] 0.417093250 0.1331148669 1.416116e-02
## [1596,] 0.433331375 0.1637029640 2.061445e-02
## [1597,] 0.163702964 0.4333313752 3.823512e-01
## [1598,] 0.416337988 0.3211750193 8.258786e-02
## [1599,] 0.440355309 0.2596967205 5.105149e-02
## [1600,] 0.416337988 0.3211750193 8.258786e-02
## [1601,] 0.433331375 0.1637029640 2.061445e-02
## [1602,] 0.335537578 0.4074384881 1.649156e-01
## [1603,] 0.443086838 0.2436977611 4.467792e-02
## [1604,] 0.440355309 0.2596967205 5.105149e-02
## [1605,] 0.386693968 0.3625255950 1.132892e-01
## [1606,] 0.291090430 0.4308138364 2.125348e-01
## [1607,] 0.148232688 0.4261689772 4.084119e-01
## [1608,] 0.360146521 0.0776786613 5.584740e-03
## [1609,] 0.440355309 0.2596967205 5.105149e-02
## [1610,] 0.243697761 0.4430868383 2.685375e-01
## [1611,] 0.426168977 0.1482326877 1.718640e-02
## [1612,] 0.430813836 0.2910904300 6.556091e-02
## [1613,] 0.407438488 0.3355375785 9.210835e-02
## [1614,] 0.397531973 0.3493462791 1.023338e-01
## [1615,] 0.416337988 0.3211750193 8.258786e-02
## [1616,] 0.426168977 0.1482326877 1.718640e-02
## [1617,] 0.406028666 0.1184250277 1.151354e-02
## [1618,] 0.291090430 0.4308138364 2.125348e-01
## [1619,] 0.169380014 0.0116813803 2.685375e-04
## [1620,] 0.426168977 0.1482326877 1.718640e-02
## [1621,] 0.386693968 0.3625255950 1.132892e-01
## [1622,] 0.375000000 0.3750000000 1.250000e-01
## [1623,] 0.397531973 0.3493462791 1.023338e-01
## [1624,] 0.433331375 0.1637029640 2.061445e-02
## [1625,] 0.362525595 0.3866939680 1.374912e-01
## [1626,] 0.291090430 0.4308138364 2.125348e-01
## [1627,] 0.416337988 0.3211750193 8.258786e-02
## [1628,] 0.443086838 0.2436977611 4.467792e-02
## [1629,] 0.397531973 0.3493462791 1.023338e-01
## [1630,] 0.436239133 0.2755194522 5.800410e-02
## [1631,] 0.386693968 0.3625255950 1.132892e-01
## [1632,] 0.375000000 0.3750000000 1.250000e-01
## [1633,] 0.349346279 0.3975319727 1.507880e-01
## [1634,] 0.243697761 0.4430868383 2.685375e-01
## [1635,] 0.406028666 0.1184250277 1.151354e-02
## [1636,] 0.291090430 0.4308138364 2.125348e-01
## [1637,] 0.266544426 0.0339238361 1.439193e-03
## [1638,] 0.033923836 0.2665444262 6.980925e-01
## [1639,] 0.000000000 0.0000000000 0.000000e+00
## [1640,] 0.335537578 0.4074384881 1.649156e-01
## [1641,] 0.349346279 0.3975319727 1.507880e-01
## [1642,] 0.424154946 0.3063341278 7.374710e-02
## [1643,] 0.360146521 0.0776786613 5.584740e-03
## [1644,] 0.386693968 0.3625255950 1.132892e-01
## [1645,] 0.179450170 0.4386559699 3.574234e-01
## [1646,] 0.236850055 0.0253767916 9.063140e-04
## [1647,] 0.386693968 0.3625255950 1.132892e-01
## [1648,] 0.306334128 0.4241549461 1.957638e-01
## [1649,] 0.386693968 0.3625255950 1.132892e-01
## [1650,] 0.033923836 0.2665444262 6.980925e-01
## [1651,] 0.377630828 0.0906313987 7.250512e-03
## [1652,] 0.386693968 0.3625255950 1.132892e-01
## [1653,] 0.360146521 0.0776786613 5.584740e-03
## [1654,] 0.443086838 0.2436977611 4.467792e-02
## [1655,] 0.335537578 0.4074384881 1.649156e-01
## [1656,] 0.407438488 0.3355375785 9.210835e-02
## [1657,] 0.424154946 0.3063341278 7.374710e-02
## [1658,] 0.443086838 0.2436977611 4.467792e-02
## [1659,] 0.392899701 0.1042386963 9.218388e-03
## [1660,] 0.046838810 0.0007678494 4.195898e-06
## [1661,] 0.430813836 0.2910904300 6.556091e-02
## [1662,] 0.275519452 0.4362391326 2.302373e-01
## [1663,] 0.291090430 0.4308138364 2.125348e-01
## [1664,] 0.436239133 0.2755194522 5.800410e-02
## [1665,] 0.318229499 0.0540389715 3.058810e-03
## [1666,] 0.426168977 0.1482326877 1.718640e-02
## [1667,] 0.397531973 0.3493462791 1.023338e-01
## [1668,] 0.417093250 0.1331148669 1.416116e-02
## [1669,] 0.433331375 0.1637029640 2.061445e-02
## [1670,] 0.443086838 0.2436977611 4.467792e-02
## [1671,] 0.397531973 0.3493462791 1.023338e-01
## [1672,] 0.416337988 0.3211750193 8.258786e-02
## [1673,] 0.306334128 0.4241549461 1.957638e-01
## [1674,] 0.440355309 0.2596967205 5.105149e-02
## [1675,] 0.407438488 0.3355375785 9.210835e-02
## [1676,] 0.424154946 0.3063341278 7.374710e-02
## [1677,] 0.424154946 0.3063341278 7.374710e-02
## [1678,] 0.407438488 0.3355375785 9.210835e-02
## [1679,] 0.444093854 0.2114732637 3.356718e-02
## [1680,] 0.417093250 0.1331148669 1.416116e-02
## [1681,] 0.335537578 0.4074384881 1.649156e-01
## [1682,] 0.417093250 0.1331148669 1.416116e-02
## [1683,] 0.406028666 0.1184250277 1.151354e-02
## [1684,] 0.444358195 0.2275981001 3.885821e-02
## [1685,] 0.438655970 0.1794501695 2.447048e-02
## [1686,] 0.442218287 0.1953987782 2.877966e-02
## [1687,] 0.443086838 0.2436977611 4.467792e-02
## [1688,] 0.275519452 0.4362391326 2.302373e-01
## [1689,] 0.375000000 0.3750000000 1.250000e-01
## [1690,] 0.406028666 0.1184250277 1.151354e-02
## [1691,] 0.386693968 0.3625255950 1.132892e-01
## [1692,] 0.386693968 0.3625255950 1.132892e-01
## [1693,] 0.406028666 0.1184250277 1.151354e-02
## [1694,] 0.377630828 0.0906313987 7.250512e-03
## [1695,] 0.417093250 0.1331148669 1.416116e-02
## [1696,] 0.275519452 0.4362391326 2.302373e-01
## [1697,] 0.407438488 0.3355375785 9.210835e-02
## [1698,] 0.375000000 0.3750000000 1.250000e-01
## [1699,] 0.442218287 0.1953987782 2.877966e-02
## [1700,] 0.321175019 0.4163379880 1.798991e-01
## [1701,] 0.275519452 0.4362391326 2.302373e-01
## [1702,] 0.275519452 0.4362391326 2.302373e-01
## [1703,] 0.386693968 0.3625255950 1.132892e-01
## [1704,] 0.397531973 0.3493462791 1.023338e-01
## [1705,] 0.335537578 0.4074384881 1.649156e-01
## [1706,] 0.443086838 0.2436977611 4.467792e-02
## [1707,] 0.433331375 0.1637029640 2.061445e-02
## [1708,] 0.443086838 0.2436977611 4.467792e-02
## [1709,] 0.169380014 0.0116813803 2.685375e-04
## [1710,] 0.386693968 0.3625255950 1.132892e-01
## [1711,] 0.443086838 0.2436977611 4.467792e-02
## [1712,] 0.416337988 0.3211750193 8.258786e-02
## [1713,] 0.377630828 0.0906313987 7.250512e-03
## [1714,] 0.407438488 0.3355375785 9.210835e-02
## [1715,] 0.406028666 0.1184250277 1.151354e-02
## [1716,] 0.321175019 0.4163379880 1.798991e-01
## [1717,] 0.406028666 0.1184250277 1.151354e-02
## [1718,] 0.444358195 0.2275981001 3.885821e-02
## [1719,] 0.349346279 0.3975319727 1.507880e-01
## [1720,] 0.443086838 0.2436977611 4.467792e-02
## [1721,] 0.118425028 0.4060286664 4.640328e-01
## [1722,] 0.443086838 0.2436977611 4.467792e-02
## [1723,] 0.335537578 0.4074384881 1.649156e-01
## [1724,] 0.406028666 0.1184250277 1.151354e-02
## [1725,] 0.416337988 0.3211750193 8.258786e-02
## [1726,] 0.442218287 0.1953987782 2.877966e-02
## [1727,] 0.375000000 0.3750000000 1.250000e-01
## [1728,] 0.321175019 0.4163379880 1.798991e-01
## [1729,] 0.118425028 0.4060286664 4.640328e-01
## [1730,] 0.440355309 0.2596967205 5.105149e-02
## [1731,] 0.306334128 0.4241549461 1.957638e-01
## [1732,] 0.236850055 0.0253767916 9.063140e-04
## [1733,] 0.179450170 0.4386559699 3.574234e-01
## [1734,] 0.163702964 0.4333313752 3.823512e-01
## [1735,] 0.293645732 0.0435030714 2.148300e-03
## [1736,] 0.416337988 0.3211750193 8.258786e-02
## [1737,] 0.204487093 0.0179374643 5.244873e-04
## [1738,] 0.392899701 0.1042386963 9.218388e-03
## [1739,] 0.430813836 0.2910904300 6.556091e-02
## [1740,] 0.386693968 0.3625255950 1.132892e-01
## [1741,] 0.291090430 0.4308138364 2.125348e-01
## [1742,] 0.386693968 0.3625255950 1.132892e-01
## [1743,] 0.163702964 0.4333313752 3.823512e-01
## [1744,] 0.259696720 0.4403553087 2.488965e-01
## [1745,] 0.077678661 0.3601465208 5.565901e-01
## [1746,] 0.392899701 0.1042386963 9.218388e-03
## [1747,] 0.444093854 0.2114732637 3.356718e-02
## [1748,] 0.424154946 0.3063341278 7.374710e-02
## [1749,] 0.392899701 0.1042386963 9.218388e-03
## [1750,] 0.375000000 0.3750000000 1.250000e-01
## [1751,] 0.293645732 0.0435030714 2.148300e-03
## [1752,] 0.377630828 0.0906313987 7.250512e-03
## [1753,] 0.443086838 0.2436977611 4.467792e-02
## [1754,] 0.424154946 0.3063341278 7.374710e-02
## [1755,] 0.133114867 0.4170932496 4.356307e-01
## [1756,] 0.306334128 0.4241549461 1.957638e-01
## [1757,] 0.275519452 0.4362391326 2.302373e-01
## [1758,] 0.442218287 0.1953987782 2.877966e-02
## [1759,] 0.407438488 0.3355375785 9.210835e-02
## [1760,] 0.442218287 0.1953987782 2.877966e-02
## [1761,] 0.243697761 0.4430868383 2.685375e-01
## [1762,] 0.349346279 0.3975319727 1.507880e-01
## [1763,] 0.436239133 0.2755194522 5.800410e-02
## [1764,] 0.407438488 0.3355375785 9.210835e-02
## [1765,] 0.430813836 0.2910904300 6.556091e-02
## [1766,] 0.397531973 0.3493462791 1.023338e-01
## [1767,] 0.424154946 0.3063341278 7.374710e-02
## [1768,] 0.438655970 0.1794501695 2.447048e-02
## [1769,] 0.360146521 0.0776786613 5.584740e-03
## [1770,] 0.090631399 0.0030210466 3.356718e-05
## [1771,] 0.406028666 0.1184250277 1.151354e-02
## [1772,] 0.438655970 0.1794501695 2.447048e-02
## [1773,] 0.392899701 0.1042386963 9.218388e-03
## [1774,] 0.340371253 0.0654560102 4.195898e-03
## [1775,] 0.436239133 0.2755194522 5.800410e-02
## [1776,] 0.148232688 0.4261689772 4.084119e-01
## [1777,] 0.442218287 0.1953987782 2.877966e-02
## [1778,] 0.377630828 0.0906313987 7.250512e-03
## [1779,] 0.293645732 0.0435030714 2.148300e-03
## [1780,] 0.424154946 0.3063341278 7.374710e-02
## [1781,] 0.386693968 0.3625255950 1.132892e-01
## [1782,] 0.321175019 0.4163379880 1.798991e-01
## [1783,] 0.436239133 0.2755194522 5.800410e-02
## [1784,] 0.266544426 0.0339238361 1.439193e-03
## [1785,] 0.335537578 0.4074384881 1.649156e-01
## [1786,] 0.444093854 0.2114732637 3.356718e-02
## [1787,] 0.360146521 0.0776786613 5.584740e-03
## [1788,] 0.259696720 0.4403553087 2.488965e-01
## [1789,] 0.362525595 0.3866939680 1.374912e-01
## [1790,] 0.204487093 0.0179374643 5.244873e-04
## [1791,] 0.195398778 0.4422182874 3.336033e-01
## [1792,] 0.065456010 0.3403712531 5.899768e-01
## [1793,] 0.227598100 0.4443581954 2.891855e-01
## [1794,] 0.266544426 0.0339238361 1.439193e-03
## [1795,] 0.386693968 0.3625255950 1.132892e-01
## [1796,] 0.335537578 0.4074384881 1.649156e-01
## [1797,] 0.424154946 0.3063341278 7.374710e-02
## [1798,] 0.430813836 0.2910904300 6.556091e-02
## [1799,] 0.349346279 0.3975319727 1.507880e-01
## [1800,] 0.430813836 0.2910904300 6.556091e-02
## [1801,] 0.340371253 0.0654560102 4.195898e-03
## [1802,] 0.306334128 0.4241549461 1.957638e-01
## [1803,] 0.438655970 0.1794501695 2.447048e-02
## [1804,] 0.054038972 0.3182294988 6.246727e-01
## [1805,] 0.204487093 0.0179374643 5.244873e-04
## [1806,] 0.436239133 0.2755194522 5.800410e-02
## [1807,] 0.318229499 0.0540389715 3.058810e-03
## [1808,] 0.360146521 0.0776786613 5.584740e-03
## [1809,] 0.440355309 0.2596967205 5.105149e-02
## [1810,] 0.169380014 0.0116813803 2.685375e-04
## [1811,] 0.444358195 0.2275981001 3.885821e-02
## [1812,] 0.375000000 0.3750000000 1.250000e-01
## [1813,] 0.436239133 0.2755194522 5.800410e-02
## [1814,] 0.291090430 0.4308138364 2.125348e-01
## [1815,] 0.397531973 0.3493462791 1.023338e-01
## [1816,] 0.377630828 0.0906313987 7.250512e-03
## [1817,] 0.275519452 0.4362391326 2.302373e-01
## [1818,] 0.430813836 0.2910904300 6.556091e-02
## [1819,] 0.433331375 0.1637029640 2.061445e-02
## [1820,] 0.243697761 0.4430868383 2.685375e-01
## [1821,] 0.077678661 0.3601465208 5.565901e-01
## [1822,] 0.090631399 0.3776308281 5.244873e-01
## [1823,] 0.335537578 0.4074384881 1.649156e-01
## [1824,] 0.118425028 0.4060286664 4.640328e-01
## [1825,] 0.377630828 0.0906313987 7.250512e-03
## [1826,] 0.430813836 0.2910904300 6.556091e-02
## [1827,] 0.306334128 0.4241549461 1.957638e-01
## [1828,] 0.442218287 0.1953987782 2.877966e-02
## [1829,] 0.407438488 0.3355375785 9.210835e-02
## [1830,] 0.321175019 0.4163379880 1.798991e-01
## [1831,] 0.392899701 0.1042386963 9.218388e-03
## [1832,] 0.000000000 0.0000000000 0.000000e+00
## [1833,] 0.375000000 0.3750000000 1.250000e-01
## [1834,] 0.443086838 0.2436977611 4.467792e-02
## [1835,] 0.433331375 0.1637029640 2.061445e-02
## [1836,] 0.407438488 0.3355375785 9.210835e-02
## [1837,] 0.443086838 0.2436977611 4.467792e-02
## [1838,] 0.444358195 0.2275981001 3.885821e-02
## [1839,] 0.436239133 0.2755194522 5.800410e-02
## [1840,] 0.442218287 0.1953987782 2.877966e-02
## [1841,] 0.243697761 0.4430868383 2.685375e-01
## [1842,] 0.443086838 0.2436977611 4.467792e-02
## [1843,] 0.318229499 0.0540389715 3.058810e-03
## [1844,] 0.392899701 0.1042386963 9.218388e-03
## [1845,] 0.424154946 0.3063341278 7.374710e-02
## [1846,] 0.444093854 0.2114732637 3.356718e-02
## [1847,] 0.426168977 0.1482326877 1.718640e-02
## [1848,] 0.440355309 0.2596967205 5.105149e-02
## [1849,] 0.090631399 0.0030210466 3.356718e-05
## [1850,] 0.444093854 0.2114732637 3.356718e-02
## [1851,] 0.430813836 0.2910904300 6.556091e-02
## [1852,] 0.362525595 0.3866939680 1.374912e-01
## [1853,] 0.291090430 0.4308138364 2.125348e-01
## [1854,] 0.236850055 0.0253767916 9.063140e-04
## [1855,] 0.440355309 0.2596967205 5.105149e-02
## [1856,] 0.442218287 0.1953987782 2.877966e-02
## [1857,] 0.436239133 0.2755194522 5.800410e-02
## [1858,] 0.266544426 0.0339238361 1.439193e-03
## [1859,] 0.416337988 0.3211750193 8.258786e-02
## [1860,] 0.443086838 0.2436977611 4.467792e-02
## [1861,] 0.430813836 0.2910904300 6.556091e-02
## [1862,] 0.362525595 0.3866939680 1.374912e-01
## [1863,] 0.436239133 0.2755194522 5.800410e-02
## [1864,] 0.046838810 0.0007678494 4.195898e-06
## [1865,] 0.424154946 0.3063341278 7.374710e-02
## [1866,] 0.293645732 0.0435030714 2.148300e-03
## [1867,] 0.306334128 0.4241549461 1.957638e-01
## [1868,] 0.406028666 0.1184250277 1.151354e-02
## [1869,] 0.375000000 0.3750000000 1.250000e-01
## [1870,] 0.433331375 0.1637029640 2.061445e-02
## [1871,] 0.426168977 0.1482326877 1.718640e-02
## [1872,] 0.204487093 0.0179374643 5.244873e-04
## [1873,] 0.211473264 0.4440938538 3.108657e-01
## [1874,] 0.397531973 0.3493462791 1.023338e-01
## [1875,] 0.386693968 0.3625255950 1.132892e-01
## [1876,] 0.433331375 0.1637029640 2.061445e-02
## [1877,] 0.291090430 0.4308138364 2.125348e-01
## [1878,] 0.433331375 0.1637029640 2.061445e-02
## [1879,] 0.442218287 0.1953987782 2.877966e-02
## [1880,] 0.318229499 0.0540389715 3.058810e-03
## [1881,] 0.148232688 0.4261689772 4.084119e-01
## [1882,] 0.293645732 0.0435030714 2.148300e-03
## [1883,] 0.440355309 0.2596967205 5.105149e-02
## [1884,] 0.169380014 0.0116813803 2.685375e-04
## [1885,] 0.407438488 0.3355375785 9.210835e-02
## [1886,] 0.204487093 0.0179374643 5.244873e-04
## [1887,] 0.424154946 0.3063341278 7.374710e-02
## [1888,] 0.090631399 0.0030210466 3.356718e-05
## [1889,] 0.430813836 0.2910904300 6.556091e-02
## [1890,] 0.407438488 0.3355375785 9.210835e-02
## [1891,] 0.417093250 0.1331148669 1.416116e-02
## [1892,] 0.179450170 0.4386559699 3.574234e-01
## [1893,] 0.444093854 0.2114732637 3.356718e-02
## [1894,] 0.407438488 0.3355375785 9.210835e-02
## [1895,] 0.163702964 0.4333313752 3.823512e-01
## [1896,] 0.243697761 0.4430868383 2.685375e-01
## [1897,] 0.204487093 0.0179374643 5.244873e-04
## [1898,] 0.362525595 0.3866939680 1.374912e-01
## [1899,] 0.433331375 0.1637029640 2.061445e-02
## [1900,] 0.444093854 0.2114732637 3.356718e-02
## [1901,] 0.438655970 0.1794501695 2.447048e-02
## [1902,] 0.406028666 0.1184250277 1.151354e-02
## [1903,] 0.440355309 0.2596967205 5.105149e-02
## [1904,] 0.293645732 0.0435030714 2.148300e-03
## [1905,] 0.293645732 0.0435030714 2.148300e-03
## [1906,] 0.266544426 0.0339238361 1.439193e-03
## [1907,] 0.243697761 0.4430868383 2.685375e-01
## [1908,] 0.259696720 0.4403553087 2.488965e-01
## [1909,] 0.377630828 0.0906313987 7.250512e-03
## [1910,] 0.424154946 0.3063341278 7.374710e-02
## [1911,] 0.360146521 0.0776786613 5.584740e-03
## [1912,] 0.349346279 0.3975319727 1.507880e-01
## [1913,] 0.442218287 0.1953987782 2.877966e-02
## [1914,] 0.104238696 0.3928997013 4.936432e-01
## [1915,] 0.426168977 0.1482326877 1.718640e-02
## [1916,] 0.362525595 0.3866939680 1.374912e-01
## [1917,] 0.444093854 0.2114732637 3.356718e-02
## [1918,] 0.291090430 0.4308138364 2.125348e-01
## [1919,] 0.444358195 0.2275981001 3.885821e-02
## [1920,] 0.306334128 0.4241549461 1.957638e-01
## [1921,] 0.375000000 0.3750000000 1.250000e-01
## [1922,] 0.444358195 0.2275981001 3.885821e-02
## [1923,] 0.406028666 0.1184250277 1.151354e-02
## [1924,] 0.397531973 0.3493462791 1.023338e-01
## [1925,] 0.443086838 0.2436977611 4.467792e-02
## [1926,] 0.349346279 0.3975319727 1.507880e-01
## [1927,] 0.340371253 0.0654560102 4.195898e-03
## [1928,] 0.291090430 0.4308138364 2.125348e-01
## [1929,] 0.424154946 0.3063341278 7.374710e-02
## [1930,] 0.377630828 0.0906313987 7.250512e-03
## [1931,] 0.443086838 0.2436977611 4.467792e-02
## [1932,] 0.375000000 0.3750000000 1.250000e-01
## [1933,] 0.430813836 0.2910904300 6.556091e-02
## [1934,] 0.424154946 0.3063341278 7.374710e-02
## [1935,] 0.406028666 0.1184250277 1.151354e-02
## [1936,] 0.426168977 0.1482326877 1.718640e-02
## [1937,] 0.438655970 0.1794501695 2.447048e-02
## [1938,] 0.349346279 0.3975319727 1.507880e-01
## [1939,] 0.211473264 0.4440938538 3.108657e-01
## [1940,] 0.438655970 0.1794501695 2.447048e-02
## [1941,] 0.440355309 0.2596967205 5.105149e-02
## [1942,] 0.275519452 0.4362391326 2.302373e-01
## [1943,] 0.424154946 0.3063341278 7.374710e-02
## [1944,] 0.416337988 0.3211750193 8.258786e-02
## [1945,] 0.266544426 0.0339238361 1.439193e-03
## [1946,] 0.335537578 0.4074384881 1.649156e-01
## [1947,] 0.377630828 0.0906313987 7.250512e-03
## [1948,] 0.360146521 0.0776786613 5.584740e-03
## [1949,] 0.204487093 0.0179374643 5.244873e-04
## [1950,] 0.386693968 0.3625255950 1.132892e-01
## [1951,] 0.424154946 0.3063341278 7.374710e-02
## [1952,] 0.349346279 0.3975319727 1.507880e-01
## [1953,] 0.438655970 0.1794501695 2.447048e-02
## [1954,] 0.204487093 0.0179374643 5.244873e-04
## [1955,] 0.349346279 0.3975319727 1.507880e-01
## [1956,] 0.397531973 0.3493462791 1.023338e-01
## [1957,] 0.426168977 0.1482326877 1.718640e-02
## [1958,] 0.426168977 0.1482326877 1.718640e-02
## [1959,] 0.430813836 0.2910904300 6.556091e-02
## [1960,] 0.430813836 0.2910904300 6.556091e-02
## [1961,] 0.227598100 0.4443581954 2.891855e-01
## [1962,] 0.321175019 0.4163379880 1.798991e-01
## [1963,] 0.090631399 0.0030210466 3.356718e-05
## [1964,] 0.443086838 0.2436977611 4.467792e-02
## [1965,] 0.386693968 0.3625255950 1.132892e-01
## [1966,] 0.430813836 0.2910904300 6.556091e-02
## [1967,] 0.275519452 0.4362391326 2.302373e-01
## [1968,] 0.291090430 0.4308138364 2.125348e-01
## [1969,] 0.444093854 0.2114732637 3.356718e-02
## [1970,] 0.335537578 0.4074384881 1.649156e-01
## [1971,] 0.443086838 0.2436977611 4.467792e-02
## [1972,] 0.360146521 0.0776786613 5.584740e-03
## [1973,] 0.444358195 0.2275981001 3.885821e-02
## [1974,] 0.362525595 0.3866939680 1.374912e-01
## [1975,] 0.362525595 0.3866939680 1.374912e-01
## [1976,] 0.259696720 0.4403553087 2.488965e-01
## [1977,] 0.377630828 0.0906313987 7.250512e-03
## [1978,] 0.275519452 0.4362391326 2.302373e-01
## [1979,] 0.104238696 0.3928997013 4.936432e-01
## [1980,] 0.349346279 0.3975319727 1.507880e-01
## [1981,] 0.416337988 0.3211750193 8.258786e-02
## [1982,] 0.306334128 0.4241549461 1.957638e-01
## [1983,] 0.204487093 0.0179374643 5.244873e-04
## [1984,] 0.025376792 0.2368500554 7.368668e-01
## [1985,] 0.442218287 0.1953987782 2.877966e-02
## [1986,] 0.291090430 0.4308138364 2.125348e-01
## [1987,] 0.266544426 0.0339238361 1.439193e-03
## [1988,] 0.118425028 0.4060286664 4.640328e-01
## [1989,] 0.163702964 0.4333313752 3.823512e-01
## [1990,] 0.424154946 0.3063341278 7.374710e-02
## [1991,] 0.406028666 0.1184250277 1.151354e-02
## [1992,] 0.430813836 0.2910904300 6.556091e-02
## [1993,] 0.442218287 0.1953987782 2.877966e-02
## [1994,] 0.293645732 0.0435030714 2.148300e-03
## [1995,] 0.444358195 0.2275981001 3.885821e-02
## [1996,] 0.416337988 0.3211750193 8.258786e-02
## [1997,] 0.443086838 0.2436977611 4.467792e-02
## [1998,] 0.349346279 0.3975319727 1.507880e-01
## [1999,] 0.430813836 0.2910904300 6.556091e-02
## [2000,] 0.335537578 0.4074384881 1.649156e-01
## [2001,] 0.362525595 0.3866939680 1.374912e-01
## [2002,] 0.306334128 0.4241549461 1.957638e-01
## [2003,] 0.340371253 0.0654560102 4.195898e-03
## [2004,] 0.340371253 0.0654560102 4.195898e-03
## [2005,] 0.293645732 0.0435030714 2.148300e-03
## [2006,] 0.416337988 0.3211750193 8.258786e-02
## [2007,] 0.033923836 0.2665444262 6.980925e-01
## [2008,] 0.392899701 0.1042386963 9.218388e-03
## [2009,] 0.443086838 0.2436977611 4.467792e-02
## [2010,] 0.444093854 0.2114732637 3.356718e-02
## [2011,] 0.436239133 0.2755194522 5.800410e-02
## [2012,] 0.362525595 0.3866939680 1.374912e-01
## [2013,] 0.349346279 0.3975319727 1.507880e-01
## [2014,] 0.443086838 0.2436977611 4.467792e-02
## [2015,] 0.266544426 0.0339238361 1.439193e-03
## [2016,] 0.397531973 0.3493462791 1.023338e-01
## [2017,] 0.104238696 0.3928997013 4.936432e-01
## [2018,] 0.424154946 0.3063341278 7.374710e-02
## [2019,] 0.417093250 0.1331148669 1.416116e-02
## [2020,] 0.360146521 0.0776786613 5.584740e-03
## [2021,] 0.318229499 0.0540389715 3.058810e-03
## [2022,] 0.443086838 0.2436977611 4.467792e-02
## [2023,] 0.438655970 0.1794501695 2.447048e-02
## [2024,] 0.386693968 0.3625255950 1.132892e-01
## [2025,] 0.321175019 0.4163379880 1.798991e-01
## [2026,] 0.444093854 0.2114732637 3.356718e-02
## [2027,] 0.065456010 0.3403712531 5.899768e-01
## [2028,] 0.236850055 0.0253767916 9.063140e-04
## [2029,] 0.169380014 0.0116813803 2.685375e-04
## [2030,] 0.360146521 0.0776786613 5.584740e-03
## [2031,] 0.444093854 0.2114732637 3.356718e-02
## [2032,] 0.054038972 0.3182294988 6.246727e-01
## [2033,] 0.406028666 0.1184250277 1.151354e-02
## [2034,] 0.406028666 0.1184250277 1.151354e-02
## [2035,] 0.417093250 0.1331148669 1.416116e-02
## [2036,] 0.438655970 0.1794501695 2.447048e-02
## [2037,] 0.407438488 0.3355375785 9.210835e-02
## [2038,] 0.227598100 0.4443581954 2.891855e-01
## [2039,] 0.377630828 0.0906313987 7.250512e-03
## [2040,] 0.306334128 0.4241549461 1.957638e-01
## [2041,] 0.392899701 0.1042386963 9.218388e-03
## [2042,] 0.426168977 0.1482326877 1.718640e-02
## [2043,] 0.397531973 0.3493462791 1.023338e-01
## [2044,] 0.360146521 0.0776786613 5.584740e-03
## [2045,] 0.243697761 0.4430868383 2.685375e-01
## [2046,] 0.440355309 0.2596967205 5.105149e-02
## [2047,] 0.275519452 0.4362391326 2.302373e-01
## [2048,] 0.335537578 0.4074384881 1.649156e-01
## [2049,] 0.321175019 0.4163379880 1.798991e-01
## [2050,] 0.442218287 0.1953987782 2.877966e-02
## [2051,] 0.433331375 0.1637029640 2.061445e-02
## [2052,] 0.443086838 0.2436977611 4.467792e-02
## [2053,] 0.306334128 0.4241549461 1.957638e-01
## [2054,] 0.442218287 0.1953987782 2.877966e-02
## [2055,] 0.444358195 0.2275981001 3.885821e-02
## [2056,] 0.397531973 0.3493462791 1.023338e-01
## [2057,] 0.349346279 0.3975319727 1.507880e-01
## [2058,] 0.397531973 0.3493462791 1.023338e-01
## [2059,] 0.340371253 0.0654560102 4.195898e-03
## [2060,] 0.133114867 0.4170932496 4.356307e-01
## [2061,] 0.436239133 0.2755194522 5.800410e-02
## [2062,] 0.243697761 0.4430868383 2.685375e-01
## [2063,] 0.375000000 0.3750000000 1.250000e-01
## [2064,] 0.424154946 0.3063341278 7.374710e-02
## [2065,] 0.386693968 0.3625255950 1.132892e-01
## [2066,] 0.436239133 0.2755194522 5.800410e-02
## [2067,] 0.377630828 0.0906313987 7.250512e-03
## [2068,] 0.392899701 0.1042386963 9.218388e-03
## [2069,] 0.360146521 0.0776786613 5.584740e-03
## [2070,] 0.442218287 0.1953987782 2.877966e-02
## [2071,] 0.275519452 0.4362391326 2.302373e-01
## [2072,] 0.424154946 0.3063341278 7.374710e-02
## [2073,] 0.266544426 0.0339238361 1.439193e-03
## [2074,] 0.392899701 0.1042386963 9.218388e-03
## [2075,] 0.349346279 0.3975319727 1.507880e-01
## [2076,] 0.266544426 0.0339238361 1.439193e-03
## [2077,] 0.362525595 0.3866939680 1.374912e-01
## [2078,] 0.377630828 0.0906313987 7.250512e-03
## [2079,] 0.443086838 0.2436977611 4.467792e-02
## [2080,] 0.426168977 0.1482326877 1.718640e-02
## [2081,] 0.436239133 0.2755194522 5.800410e-02
## [2082,] 0.377630828 0.0906313987 7.250512e-03
## [2083,] 0.293645732 0.0435030714 2.148300e-03
## [2084,] 0.360146521 0.0776786613 5.584740e-03
## [2085,] 0.306334128 0.4241549461 1.957638e-01
## [2086,] 0.349346279 0.3975319727 1.507880e-01
## [2087,] 0.375000000 0.3750000000 1.250000e-01
## [2088,] 0.321175019 0.4163379880 1.798991e-01
## [2089,] 0.443086838 0.2436977611 4.467792e-02
## [2090,] 0.335537578 0.4074384881 1.649156e-01
## [2091,] 0.275519452 0.4362391326 2.302373e-01
## [2092,] 0.377630828 0.0906313987 7.250512e-03
## [2093,] 0.349346279 0.3975319727 1.507880e-01
## [2094,] 0.406028666 0.1184250277 1.151354e-02
## [2095,] 0.362525595 0.3866939680 1.374912e-01
## [2096,] 0.293645732 0.0435030714 2.148300e-03
## [2097,] 0.392899701 0.1042386963 9.218388e-03
## [2098,] 0.392899701 0.1042386963 9.218388e-03
## [2099,] 0.424154946 0.3063341278 7.374710e-02
## [2100,] 0.377630828 0.0906313987 7.250512e-03
## [2101,] 0.318229499 0.0540389715 3.058810e-03
## [2102,] 0.291090430 0.4308138364 2.125348e-01
## attr(,"degree")
## [1] 3
## attr(,"knots")
## numeric(0)
## attr(,"Boundary.knots")
## [1] 18 80
## attr(,"intercept")
## [1] FALSE
## attr(,"class")
## [1] "bs"     "basis"  "matrix"
```
##Notes and further reading

* Level 1 frature creation (raw data to covariates)
    * Science is key. Google "feature extraction for [data type]"
    * Err on overcreation of features
    * in some applications (images, voices) automated feature creation is possible/necessary
    * *that link from the lecture*
* Level 2 feature creation (covariates to new covariates)
    * The function `preProcess` in `caret` will handle some preprocessing
    * Create new covariates if you think they will improve the fit
    * Use exploratory analysis on the training set for creating them
    * Be careful about overfitting!
* If you want tot fit spline models, use the `gam` method in the `caret` package which allows smoothing of multiple variables
* More on feature creation/data tidying in the Obtaining Data course from the Data Science (done already)

#Preprocessing with principle components analysis


```r
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

M<- abs(cor(training[,-58]))
diag(M) <-0
which(M>0.8, arr.ind=T)
```

```
##        row col
## num415  34  32
## direct  40  32
## num857  32  34
## direct  40  34
## num857  32  40
## num415  34  40
```

```r
names(spam)[c(34, 32)]
```

```
## [1] "num415" "num857"
```

```r
plot(spam[, 34], spam[,32])
```

![](lecture_2_files/figure-html/unnamed-chunk-9-1.png) 

##Basic PCA idea

* We might not need every predictor
* A weighted combination of predictors might be better
* We should pick this combination to capture the "most information" possible
* Benefits
    * Reduced number of predictors
    * Reduced noise (due to averaging)


```r
X<- 0.71*training$num415 + 0.71* training$num857
Y<- 0.71*training$num415 - 0.71* training$num857
plot(X, Y)
```

![](lecture_2_files/figure-html/unnamed-chunk-10-1.png) 

##Related problems

You have multivariate X1 ... Xn so X1=(X11 ... Xm)

* Find a new set of multivariate variables that are uncorrelated and explain as much variace as possible
* If you put all the variables together in one matrix, find the best matrix created with fewer varibales (lower rank) that explains the original data

The first goal is statistical and the second is data compression

###SVD

If X is a matrix with each variable in a column and each observation is a row then the SVD is a "matrix decompression"

X=UDV#T

where the column of U are orthogonal (left singular vectors), the columns of V are orthogonal (right singular vectors) and D is a diagonal matrix (singular values)

###PCA

The principal components are equal to the right singular values if you first scale (substract the mean, devide by standard deviation) the variables.


```r
smallSpam <- spam[,c(34, 32)]
prComp<- prcomp(smallSpam)
plot(prComp$x[,1], prComp$x[,2])
```

![](lecture_2_files/figure-html/unnamed-chunk-11-1.png) 

```r
prComp$rotation
```

```
##              PC1        PC2
## num415 0.7080625  0.7061498
## num857 0.7061498 -0.7080625
```

```r
typeColor <- ((spam$type=="spam")*1+1)
prComp<- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1], prComp$x[,2], col=typeColor, xlab="PC1", ylab="PC2")
```

![](lecture_2_files/figure-html/unnamed-chunk-11-2.png) 

```r
preProc <- preProcess(log10(spam[,-58]+1), method="pca", pcaComp = 2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[, 1], spamPC[,2], col=typeColor)
```

![](lecture_2_files/figure-html/unnamed-chunk-11-3.png) 

```r
trainPC <- predict(preProc, log10(training[, -58]+1))
modelFit<- train(training$type ~., method="glm", data=trainPC)
testPC<- predict(preProc, log10(testing[, -58]+1))
confusionMatrix(testing$type, predict(modelFit, testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction nonspam spam
##    nonspam     644   53
##    spam         73  380
##                                           
##                Accuracy : 0.8904          
##                  95% CI : (0.8709, 0.9079)
##     No Information Rate : 0.6235          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.7688          
##  Mcnemar's Test P-Value : 0.09052         
##                                           
##             Sensitivity : 0.8982          
##             Specificity : 0.8776          
##          Pos Pred Value : 0.9240          
##          Neg Pred Value : 0.8389          
##              Prevalence : 0.6235          
##          Detection Rate : 0.5600          
##    Detection Prevalence : 0.6061          
##       Balanced Accuracy : 0.8879          
##                                           
##        'Positive' Class : nonspam         
## 
```

```r
modelFit<- train(training$type ~., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction nonspam spam
##    nonspam     658   39
##    spam         50  403
##                                           
##                Accuracy : 0.9226          
##                  95% CI : (0.9056, 0.9374)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8372          
##  Mcnemar's Test P-Value : 0.2891          
##                                           
##             Sensitivity : 0.9294          
##             Specificity : 0.9118          
##          Pos Pred Value : 0.9440          
##          Neg Pred Value : 0.8896          
##              Prevalence : 0.6157          
##          Detection Rate : 0.5722          
##    Detection Prevalence : 0.6061          
##       Balanced Accuracy : 0.9206          
##                                           
##        'Positive' Class : nonspam         
## 
```

* Most useful for linear-type models
* Can make it harder to interpret predictors
* Watch for ouliers!
    * Transform first (with logs/BoxCox)
    * Plot predictors to identify problems
For more see the "Elements of Statistical Learning" (have started reading, stopped at logistic regression)

#Predicting with regression

* fit a simple model
* plug in new covariates and multiply by the coefficients
* useful when the linear model is (nearly correct)

Pros: 

* Easy to implement
* Easy to interpret

Cons:

* Often poor performance in nonlinear settings


```r
library(caret)
data("faithful")
set.seed(333)

inTrain<- createDataPartition(y=faithful$waiting, p=0.5, list=FALSE)
trainFaith<- faithful[inTrain,]
testFaith<- faithful[-inTrain,]
head(trainFaith)
```

```
##   eruptions waiting
## 1     3.600      79
## 3     3.333      74
## 5     4.533      85
## 6     2.883      55
## 7     4.700      88
## 8     3.600      85
```

```r
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")

lm1<- lm(eruptions ~ waiting, data=trainFaith)
summary(lm1)
```

```
## 
## Call:
## lm(formula = eruptions ~ waiting, data = trainFaith)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -1.26990 -0.34789  0.03979  0.36589  1.05020 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -1.792739   0.227869  -7.867 1.04e-12 ***
## waiting      0.073901   0.003148  23.474  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.495 on 135 degrees of freedom
## Multiple R-squared:  0.8032,	Adjusted R-squared:  0.8018 
## F-statistic:   551 on 1 and 135 DF,  p-value: < 2.2e-16
```

```r
lines(trainFaith$waiting, lm1$fitted.values, lwd=3)
```

![](lecture_2_files/figure-html/unnamed-chunk-12-1.png) 

```r
coef(lm1)[1] + coef(lm1)[2]*0.8
```

```
## (Intercept) 
##   -1.733619
```

```r
newdata<- data.frame(waiting=80)
predict(lm1, newdata)
```

```
##        1 
## 4.119307
```

```r
par(mfrow=c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
lines(trainFaith$waiting, predict(lm1), lwd=3)
plot(testFaith$waiting, testFaith$eruptions,pch=19, col="blue", xlab="Waiting", ylab="Duration")
lines(testFaith$waiting, predict(lm1, newdata=testFaith), lwd=3)
```

![](lecture_2_files/figure-html/unnamed-chunk-12-2.png) 

```r
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2))
```

```
## [1] 5.75186
```

```r
sqrt(sum((predict(lm1, newdata=testFaith)-testFaith$eruptions)^2))
```

```
## [1] 5.838559
```

```r
pred1 <- predict(lm1, newdata = testFaith, interval="prediction")
ord<- order(testFaith$waiting)
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue")
matlines(testFaith$waiting[ord], pred1[ord,], type="l", col=c(1,2,2), lty=c(1,1,1), lwd=3)

modFit<- train(eruptions~waiting, data=trainFaith, method="lm")
summary(modFit$finalModel)
```

```
## 
## Call:
## lm(formula = .outcome ~ ., data = dat)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -1.26990 -0.34789  0.03979  0.36589  1.05020 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -1.792739   0.227869  -7.867 1.04e-12 ***
## waiting      0.073901   0.003148  23.474  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.495 on 135 degrees of freedom
## Multiple R-squared:  0.8032,	Adjusted R-squared:  0.8018 
## F-statistic:   551 on 1 and 135 DF,  p-value: < 2.2e-16
```

![](lecture_2_files/figure-html/unnamed-chunk-12-3.png) 

* Regression models with multiple covariates can be included
* Often useful in combination with other models

#Prediction with regression, multiple covariates


```r
library(ISLR)
library(ggplot2)
library(caret)
data("Wage")
Wage<- subset(Wage, select=-c(logwage))
summary(Wage)
```

```
##       year           age               sex                    maritl    
##  Min.   :2003   Min.   :18.00   1. Male  :3000   1. Never Married: 648  
##  1st Qu.:2004   1st Qu.:33.75   2. Female:   0   2. Married      :2074  
##  Median :2006   Median :42.00                    3. Widowed      :  19  
##  Mean   :2006   Mean   :42.41                    4. Divorced     : 204  
##  3rd Qu.:2008   3rd Qu.:51.00                    5. Separated    :  55  
##  Max.   :2009   Max.   :80.00                                           
##                                                                         
##        race                   education                     region    
##  1. White:2480   1. < HS Grad      :268   2. Middle Atlantic   :3000  
##  2. Black: 293   2. HS Grad        :971   1. New England       :   0  
##  3. Asian: 190   3. Some College   :650   3. East North Central:   0  
##  4. Other:  37   4. College Grad   :685   4. West North Central:   0  
##                  5. Advanced Degree:426   5. South Atlantic    :   0  
##                                           6. East South Central:   0  
##                                           (Other)              :   0  
##            jobclass               health      health_ins  
##  1. Industrial :1544   1. <=Good     : 858   1. Yes:2083  
##  2. Information:1456   2. >=Very Good:2142   2. No : 917  
##                                                           
##                                                           
##                                                           
##                                                           
##                                                           
##       wage       
##  Min.   : 20.09  
##  1st Qu.: 85.38  
##  Median :104.92  
##  Mean   :111.70  
##  3rd Qu.:128.68  
##  Max.   :318.34  
## 
```

```r
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training<- Wage[inTrain,]
testing<- Wage[-inTrain,]
dim(training)
```

```
## [1] 2102   11
```

```r
dim(testing)
```

```
## [1] 898  11
```

```r
featurePlot(x=training[, c("age", "education", "jobclass")],
            y=training$wage, plot="pairs")
```

![](lecture_2_files/figure-html/unnamed-chunk-13-1.png) 

```r
qplot(age, wage, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-2.png) 

```r
qplot(age, wage, colour=jobclass, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-3.png) 

```r
qplot(age, wage, colour=education, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-4.png) 

```r
modFit<- train(wage~age+jobclass+education, method="lm", data=training)
finMod<- modFit$finalModel
print(modFit)
```

```
## Linear Regression 
## 
## 2102 samples
##   10 predictor
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 2102, 2102, 2102, 2102, 2102, 2102, ... 
## Resampling results
## 
##   RMSE      Rsquared   RMSE SD   Rsquared SD
##   34.71874  0.2557389  1.608216  0.02112529 
## 
## 
```

```r
plot(finMod, 1, pch=19, cex=0.5, col="#00000010")
```

![](lecture_2_files/figure-html/unnamed-chunk-13-5.png) 

```r
qplot(finMod$fitted.values, finMod$residuals, colour=race, data=training)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-6.png) 

```r
plot(finMod$residuals, pch=19) #some variables are missing from the model and/or data
```

![](lecture_2_files/figure-html/unnamed-chunk-13-7.png) 

```r
pred<- predict(modFit, testing)
qplot(wage, pred, colour=year, data=testing)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-8.png) 

```r
modFitAll<- train(wage~., data=training, method="lm")
pred<- predict(modFitAll, testing)
qplot(wage, pred, data=testing)
```

![](lecture_2_files/figure-html/unnamed-chunk-13-9.png) 

* Often use in combination with other models
* Read stuff on the above







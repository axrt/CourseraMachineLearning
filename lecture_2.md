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
##        -1.769e+00         -3.424e-01         -1.274e-01  
##               all              num3d                our  
##         1.984e-01          2.597e+00          4.752e-01  
##              over             remove           internet  
##         9.214e-01          2.558e+00          5.796e-01  
##             order               mail            receive  
##         7.853e-01          2.035e-01         -8.889e-01  
##              will             people             report  
##        -1.639e-01          4.334e-02          7.991e-02  
##         addresses               free           business  
##         1.029e+00          1.032e+00          8.628e-01  
##             email                you             credit  
##         1.655e-01          5.928e-02          1.964e+00  
##              your               font             num000  
##         3.191e-01          9.386e-02          3.320e+00  
##             money                 hp                hpl  
##         3.229e-01         -1.667e+00         -9.410e-01  
##            george             num650                lab  
##        -2.145e+01          4.734e-01         -4.179e+00  
##              labs             telnet             num857  
##        -6.733e-01         -9.057e-02          7.184e+00  
##              data             num415              num85  
##        -5.749e-01          5.376e-03         -1.901e+00  
##        technology            num1999              parts  
##         8.838e-01          9.151e-02         -6.087e-01  
##                pm             direct                 cs  
##        -6.604e-01         -4.229e-01         -5.483e+01  
##           meeting           original            project  
##        -3.347e+00         -2.990e+00         -2.128e+00  
##                re                edu              table  
##        -8.070e-01         -1.668e+00         -2.551e+00  
##        conference      charSemicolon   charRoundbracket  
##        -3.559e+00         -1.060e+00         -3.430e-01  
## charSquarebracket    charExclamation         charDollar  
##        -6.210e-01          5.645e-01          5.238e+00  
##          charHash         capitalAve        capitalLong  
##         2.327e+00          7.120e-02          7.658e-03  
##      capitalTotal  
##         9.562e-04  
## 
## Degrees of Freedom: 3450 Total (i.e. Null);  3393 Residual
## Null Deviance:	    4628 
## Residual Deviance: 1300 	AIC: 1416
```

```r
predictions <- predict(modelFit, newdata=testing)
predictions
```

```
##    [1] spam    spam    spam    nonspam spam    nonspam spam    spam   
##    [9] spam    nonspam spam    spam    spam    nonspam spam    spam   
##   [17] spam    spam    spam    spam    nonspam nonspam spam    nonspam
##   [25] spam    spam    spam    nonspam spam    nonspam spam    spam   
##   [33] spam    spam    spam    spam    spam    nonspam spam    spam   
##   [41] spam    spam    spam    spam    nonspam spam    spam    spam   
##   [49] spam    spam    spam    spam    spam    spam    spam    spam   
##   [57] spam    spam    spam    spam    spam    spam    spam    spam   
##   [65] spam    spam    spam    nonspam spam    spam    spam    spam   
##   [73] spam    nonspam nonspam spam    spam    spam    nonspam spam   
##   [81] spam    spam    spam    spam    spam    spam    spam    nonspam
##   [89] spam    spam    spam    spam    spam    spam    spam    spam   
##   [97] nonspam spam    spam    spam    spam    spam    spam    spam   
##  [105] spam    spam    spam    spam    spam    spam    spam    spam   
##  [113] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [121] spam    spam    spam    spam    spam    spam    spam    spam   
##  [129] spam    spam    spam    spam    spam    spam    spam    spam   
##  [137] nonspam nonspam spam    spam    spam    spam    spam    spam   
##  [145] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [153] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [161] spam    spam    spam    spam    spam    spam    spam    spam   
##  [169] spam    spam    spam    spam    spam    spam    spam    spam   
##  [177] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [185] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [193] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [201] spam    spam    spam    spam    spam    spam    spam    spam   
##  [209] spam    spam    spam    spam    spam    spam    spam    spam   
##  [217] spam    spam    spam    spam    spam    spam    spam    spam   
##  [225] spam    spam    spam    spam    spam    spam    spam    spam   
##  [233] spam    spam    spam    spam    nonspam nonspam spam    spam   
##  [241] spam    spam    spam    spam    spam    spam    spam    spam   
##  [249] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [257] spam    spam    spam    spam    spam    spam    spam    spam   
##  [265] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [273] spam    spam    spam    spam    spam    spam    spam    spam   
##  [281] spam    spam    spam    spam    spam    spam    spam    spam   
##  [289] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [297] nonspam spam    spam    spam    spam    spam    spam    spam   
##  [305] spam    spam    spam    spam    spam    spam    spam    spam   
##  [313] nonspam spam    spam    spam    nonspam spam    spam    nonspam
##  [321] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [329] spam    spam    spam    spam    spam    spam    spam    spam   
##  [337] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [345] spam    spam    spam    spam    spam    nonspam spam    spam   
##  [353] spam    spam    spam    spam    spam    spam    spam    spam   
##  [361] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [369] nonspam nonspam spam    spam    nonspam spam    spam    spam   
##  [377] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [385] spam    spam    spam    spam    spam    spam    spam    spam   
##  [393] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [401] spam    nonspam nonspam nonspam spam    nonspam spam    spam   
##  [409] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [417] spam    spam    spam    spam    nonspam spam    spam    nonspam
##  [425] spam    nonspam spam    spam    spam    spam    nonspam spam   
##  [433] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [441] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [449] spam    spam    nonspam spam    spam    nonspam nonspam nonspam
##  [457] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [465] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [473] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [481] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [489] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [497] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [505] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [513] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [521] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [529] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [537] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [545] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [553] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [561] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [569] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [577] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [585] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [593] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [601] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [609] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [617] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [625] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [633] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [641] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [649] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [657] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [665] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [673] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [681] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [689] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [697] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [705] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [713] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [721] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [729] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [737] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [745] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [753] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [761] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [769] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [777] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [785] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [793] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [801] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [809] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [817] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [825] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [833] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
##  [841] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [849] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [857] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [865] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [873] nonspam spam    nonspam nonspam nonspam nonspam spam    nonspam
##  [881] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [889] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [897] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [905] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [913] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [921] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [929] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [937] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [945] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [953] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [961] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [969] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [977] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [985] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [993] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1001] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1009] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1017] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1025] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1033] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1041] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
## [1049] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1057] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
## [1065] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1073] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
## [1081] nonspam nonspam spam    spam    nonspam nonspam nonspam nonspam
## [1089] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1097] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
## [1105] nonspam nonspam spam    nonspam nonspam spam    nonspam nonspam
## [1113] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
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
##    nonspam     667   55
##    spam         30  398
##                                           
##                Accuracy : 0.9261          
##                  95% CI : (0.9094, 0.9405)
##     No Information Rate : 0.6061          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8437          
##  Mcnemar's Test P-Value : 0.009237        
##                                           
##             Sensitivity : 0.9570          
##             Specificity : 0.8786          
##          Pos Pred Value : 0.9238          
##          Neg Pred Value : 0.9299          
##              Prevalence : 0.6061          
##          Detection Rate : 0.5800          
##    Detection Prevalence : 0.6278          
##       Balanced Accuracy : 0.9178          
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

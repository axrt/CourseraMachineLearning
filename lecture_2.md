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
##         -1.786686          -0.370309          -0.130948  
##               all              num3d                our  
##          0.029286           3.209758           0.513960  
##              over             remove           internet  
##          1.115030           2.452154           0.450989  
##             order               mail            receive  
##          0.445195           0.096226          -0.162999  
##              will             people             report  
##         -0.149308          -0.039720           0.175742  
##         addresses               free           business  
##          1.526551           0.979944           0.816495  
##             email                you             credit  
##          0.246962           0.074657           0.826884  
##              your               font             num000  
##          0.301445           0.198857           2.095391  
##             money                 hp                hpl  
##          0.347769          -1.563455          -1.310579  
##            george             num650                lab  
##        -10.333233           0.956098          -3.611063  
##              labs             telnet             num857  
##         -0.236730          -0.113959           2.725156  
##              data             num415              num85  
##         -0.581985          -0.251401          -2.551405  
##        technology            num1999              parts  
##          0.842258           0.024245          -0.567087  
##                pm             direct                 cs  
##         -1.384041          -0.368643         -50.408968  
##           meeting           original            project  
##         -2.591900          -0.980488          -1.419843  
##                re                edu              table  
##         -0.603823          -1.310905          -3.166888  
##        conference      charSemicolon   charRoundbracket  
##         -3.171703          -1.176734          -0.225122  
## charSquarebracket    charExclamation         charDollar  
##         -0.467083           1.085912           5.254714  
##          charHash         capitalAve        capitalLong  
##          2.107146          -0.001083           0.009495  
##      capitalTotal  
##          0.001021  
## 
## Degrees of Freedom: 3450 Total (i.e. Null);  3393 Residual
## Null Deviance:	    4628 
## Residual Deviance: 1318 	AIC: 1434
```

```r
predictions <- predict(modelFit, newdata=testing)
predictions
```

```
##    [1] spam    spam    spam    spam    spam    spam    nonspam spam   
##    [9] spam    spam    spam    nonspam spam    spam    spam    spam   
##   [17] spam    spam    spam    spam    spam    nonspam spam    spam   
##   [25] spam    spam    spam    spam    spam    spam    spam    spam   
##   [33] spam    spam    spam    spam    spam    spam    spam    spam   
##   [41] spam    spam    nonspam spam    spam    spam    spam    spam   
##   [49] spam    spam    spam    spam    spam    spam    spam    spam   
##   [57] spam    spam    spam    spam    spam    spam    spam    spam   
##   [65] spam    spam    nonspam nonspam nonspam spam    spam    spam   
##   [73] spam    nonspam spam    nonspam spam    spam    spam    spam   
##   [81] spam    spam    spam    spam    spam    spam    spam    spam   
##   [89] spam    spam    spam    spam    spam    spam    spam    spam   
##   [97] spam    spam    spam    nonspam spam    spam    nonspam spam   
##  [105] spam    spam    spam    spam    spam    spam    spam    spam   
##  [113] spam    nonspam spam    spam    spam    spam    spam    spam   
##  [121] spam    spam    spam    spam    spam    spam    spam    spam   
##  [129] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [137] spam    spam    spam    spam    spam    nonspam spam    spam   
##  [145] spam    spam    spam    spam    spam    spam    nonspam spam   
##  [153] nonspam spam    spam    spam    spam    spam    spam    spam   
##  [161] spam    spam    spam    spam    spam    spam    spam    spam   
##  [169] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [177] spam    spam    spam    spam    spam    spam    spam    spam   
##  [185] spam    spam    spam    spam    spam    spam    spam    spam   
##  [193] spam    spam    spam    spam    spam    spam    spam    spam   
##  [201] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [209] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [217] spam    spam    spam    spam    spam    spam    spam    spam   
##  [225] spam    spam    spam    spam    spam    spam    spam    spam   
##  [233] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [241] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [249] spam    spam    spam    spam    spam    spam    spam    spam   
##  [257] spam    spam    spam    spam    spam    spam    spam    nonspam
##  [265] spam    spam    spam    spam    spam    spam    spam    spam   
##  [273] spam    spam    spam    nonspam spam    spam    spam    spam   
##  [281] spam    spam    spam    spam    spam    spam    spam    spam   
##  [289] spam    spam    spam    spam    spam    spam    spam    spam   
##  [297] spam    spam    spam    spam    spam    spam    spam    spam   
##  [305] spam    spam    spam    spam    spam    spam    spam    spam   
##  [313] spam    nonspam spam    spam    spam    spam    nonspam spam   
##  [321] spam    spam    spam    spam    spam    spam    spam    spam   
##  [329] spam    spam    nonspam spam    spam    spam    spam    spam   
##  [337] spam    spam    spam    spam    spam    spam    spam    spam   
##  [345] spam    spam    spam    spam    spam    spam    spam    spam   
##  [353] spam    spam    spam    spam    nonspam spam    spam    spam   
##  [361] spam    spam    spam    spam    spam    spam    spam    spam   
##  [369] spam    spam    spam    spam    spam    spam    spam    spam   
##  [377] spam    spam    nonspam spam    spam    spam    nonspam spam   
##  [385] spam    spam    spam    spam    spam    nonspam spam    spam   
##  [393] nonspam nonspam spam    spam    spam    spam    nonspam spam   
##  [401] spam    spam    spam    spam    spam    spam    spam    spam   
##  [409] spam    spam    spam    spam    nonspam spam    spam    nonspam
##  [417] spam    spam    spam    spam    spam    nonspam nonspam spam   
##  [425] spam    nonspam nonspam spam    spam    spam    nonspam spam   
##  [433] spam    nonspam nonspam nonspam spam    spam    nonspam nonspam
##  [441] spam    nonspam spam    spam    nonspam spam    spam    spam   
##  [449] spam    spam    spam    spam    spam    nonspam nonspam nonspam
##  [457] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [465] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [473] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [481] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [489] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [497] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [505] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [513] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [521] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [529] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [537] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [545] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
##  [553] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [561] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [569] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
##  [577] nonspam nonspam spam    nonspam nonspam nonspam nonspam nonspam
##  [585] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [593] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [601] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [609] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [617] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [625] nonspam nonspam nonspam spam    nonspam spam    nonspam nonspam
##  [633] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [641] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [649] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [657] nonspam nonspam nonspam nonspam nonspam nonspam spam    nonspam
##  [665] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [673] nonspam nonspam nonspam spam    nonspam nonspam nonspam nonspam
##  [681] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [689] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [697] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [705] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [713] nonspam nonspam nonspam spam    nonspam spam    spam    nonspam
##  [721] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [729] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [737] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [745] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [753] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [761] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [769] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [777] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [785] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [793] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [801] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [809] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [817] nonspam nonspam nonspam nonspam nonspam nonspam nonspam spam   
##  [825] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [833] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [841] nonspam nonspam nonspam spam    nonspam nonspam nonspam nonspam
##  [849] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [857] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [865] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
##  [873] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [881] spam    spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [889] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [897] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [905] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [913] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [921] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [929] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [937] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
##  [945] nonspam spam    nonspam nonspam nonspam nonspam nonspam nonspam
##  [953] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [961] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [969] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [977] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [985] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
##  [993] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1001] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1009] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1017] nonspam nonspam nonspam spam    spam    nonspam spam    nonspam
## [1025] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1033] nonspam nonspam nonspam nonspam spam    spam    nonspam nonspam
## [1041] nonspam nonspam nonspam nonspam spam    spam    nonspam nonspam
## [1049] nonspam nonspam nonspam nonspam nonspam spam    nonspam nonspam
## [1057] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1065] nonspam nonspam nonspam nonspam spam    nonspam nonspam nonspam
## [1073] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1081] spam    nonspam nonspam nonspam nonspam nonspam nonspam spam   
## [1089] spam    nonspam nonspam nonspam nonspam nonspam nonspam nonspam
## [1097] nonspam nonspam spam    nonspam spam    spam    spam    spam   
## [1105] nonspam spam    nonspam nonspam spam    spam    nonspam nonspam
## [1113] nonspam nonspam nonspam nonspam nonspam nonspam nonspam nonspam
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
##    nonspam     651   47
##    spam         46  406
##                                           
##                Accuracy : 0.9191          
##                  95% CI : (0.9018, 0.9342)
##     No Information Rate : 0.6061          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8306          
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.9340          
##             Specificity : 0.8962          
##          Pos Pred Value : 0.9327          
##          Neg Pred Value : 0.8982          
##              Prevalence : 0.6061          
##          Detection Rate : 0.5661          
##    Detection Prevalence : 0.6070          
##       Balanced Accuracy : 0.9151          
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



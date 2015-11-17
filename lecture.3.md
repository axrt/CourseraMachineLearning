# Lecture 3
Alexander Tuzhikov  
November 16, 2015  

#Predicting with trees

* Iteratively split variables into groups
* Evaluate "homogenity" within each group
* Split again if necessary

##Pros:

* Easy to interpret
* Better performance in nonlinear settings

##Cons:

* Without pruning/cross-validation can lead to overfitting
* Harder to estimate uncertaintly
* Results may be variable

##Algorithm

1. Start with all variables in one group
2. Find the variable/split that best separates the outcomes
3. Divide the data into two groups ("leaves") on that split ("node")
4. Within each split, find the best variable/split that separates the outcomes
5. Continue until the groups are too small or sufficiently "pure"


```r
data("iris")
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
names(iris)
```

```
## [1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width" 
## [5] "Species"
```

```r
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

```r
inTrain<- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training<- iris[inTrain,]
testing<- iris[-inTrain,]
dim(training)
```

```
## [1] 105   5
```

```r
dim(testing)
```

```
## [1] 45  5
```

```r
qplot(Petal.Width, Sepal.Width, colour=Species, data=training)
```

![](lecture.3_files/figure-html/unnamed-chunk-1-1.png) 

```r
modFit<- train(Species ~ ., method="rpart", data=training)
```

```
## Loading required package: rpart
```

```r
print(modFit$finalModel)
```

```
## n= 105 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 105 70 setosa (0.3333333 0.3333333 0.3333333)  
##   2) Petal.Length< 2.35 35  0 setosa (1.0000000 0.0000000 0.0000000) *
##   3) Petal.Length>=2.35 70 35 versicolor (0.0000000 0.5000000 0.5000000)  
##     6) Petal.Width< 1.75 38  4 versicolor (0.0000000 0.8947368 0.1052632) *
##     7) Petal.Width>=1.75 32  1 virginica (0.0000000 0.0312500 0.9687500) *
```

```r
plot(modFit$finalModel, uniform=TRUE, main= "Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
```

![](lecture.3_files/figure-html/unnamed-chunk-1-2.png) 

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.0.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
library(rpart.plot)
predict(modFit, newdata=testing)
```

```
##  [1] setosa     setosa     setosa     setosa     setosa     setosa    
##  [7] setosa     setosa     setosa     setosa     setosa     setosa    
## [13] setosa     setosa     setosa     versicolor versicolor versicolor
## [19] versicolor versicolor versicolor versicolor versicolor versicolor
## [25] versicolor versicolor versicolor versicolor versicolor versicolor
## [31] virginica  virginica  virginica  virginica  virginica  virginica 
## [37] versicolor virginica  virginica  virginica  virginica  virginica 
## [43] virginica  virginica  virginica 
## Levels: setosa versicolor virginica
```

```r
fancyRpartPlot(modFit$finalModel)
```

![](lecture.3_files/figure-html/unnamed-chunk-1-3.png) 

* Classification trees are non-linear models
    * they use interactions between variables
    * data transformations my be less importatant (monotone transformations)
    * Threes can also be used for regression problems (continous outcome)
* Note that there are multiple tree building options in R both in the `caret` package - `part`, `rpart` and out of the `caret` package - `tree`

Book: "Calssification and regression trees"

#Bagging (bootstrap aggregating)

##Basic idea:

1. Take your dataset and resample cases, recalculate predictions
2. Average or majority vote

##Notes:

* Similar bias
* Reduced variance
* More useful for non-linear functions


```r
library(ElemStatLearn)
data("ozone",package = "ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)
```

```
##     ozone radiation temperature wind
## 17      1         8          59  9.7
## 19      4        25          61  9.7
## 14      6        78          57 18.4
## 45      7        48          80 14.3
## 106     7        49          69 10.3
## 7       8        19          61 20.1
```

```r
ll<- matrix(NA, nrow=10, ncol=155)
for(i in 1:10){
        ss<- sample(1:dim(ozone)[1], replace=TRUE)
        ozone0<- ozone[ss,]
        ozone0<- ozone0[order(ozone0$ozone),]
        loess0<- loess(temperature ~ ozone, data=ozone0, span=0.2)
        ll[i,]<- predict(loess0, newdata=data.frame(ozone=1:155))
}

plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){
        lines(1:155, ll[i,], col="grey", lwd=2)
}
lines(1:155, apply(ll, 2, mean), col="red", lwd=2)
```

![](lecture.3_files/figure-html/unnamed-chunk-2-1.png) 

##Bagging in caret

* Some models perform bagging for you, in `train` function consider `method` options:
    * `bagEarth`
    * `treebag`
    * `bagFDA`
* Alternatively you can bag any model you choose using the `bag` function

##Creating a roll-your-own


```r
predictors <- data.frame(ozone=ozone$ozone)
temperature<- ozone$temperature
treebag <- bag(predictors, temperature, B=10, bagControl = bagControl(fit=ctreeBag$fit, predict = ctreeBag$pred, aggregate = ctreeBag$aggregate))
```

[source](http://www.inside-r.org/packages/cran/docs/nbBag)


```r
plot(ozone$ozone, temperature, col="lightgrey", pch=19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), pch=19, col="red")
points(ozone$ozone, predict(treebag, predictors), pch=19, col="blue")
```

![](lecture.3_files/figure-html/unnamed-chunk-4-1.png) 

```r
ctreeBag$fit
```

```
## function (x, y, ...) 
## {
##     loadNamespace("party")
##     data <- as.data.frame(x)
##     data$y <- y
##     party::ctree(y ~ ., data = data)
## }
## <environment: namespace:caret>
```

##Notes:

* Bagging is most useful for nonlinear models
* Often used with trees - an extension is random forests
* Several models use bagging in `caret`'s `train` function

Read the links about Bagging and Boosting

#Random Forests

1. Bootstrap samples
2. At each split, bootstrap variables
3. Grow multiple trees and vote

##Pros

1. Accuracy

##Cons

1. Speed
2. Interpretability
3. Overfitting


```r
data(iris)
library(ggplot2)

inTrain<- createDataPartition(y=iris$Species, p=0.7, list=FALSE)

training<- iris[inTrain,]
testing<- iris[-inTrain,]

library(caret)
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit <- train(Species~.,data=training, method="rf", prox=TRUE)
modFit
```

```
## Random Forest 
## 
## 105 samples
##   4 predictor
##   3 classes: 'setosa', 'versicolor', 'virginica' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 105, 105, 105, 105, 105, 105, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##   2     0.9629572  0.9436810  0.03491130   0.05259170
##   3     0.9639829  0.9452891  0.03405705   0.05117056
##   4     0.9617573  0.9418867  0.03694159   0.05565854
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 3.
```

```r
getTree(modFit$finalModel,k=2)
```

```
##    left daughter right daughter split var split point status prediction
## 1              2              3         3        2.60      1          0
## 2              0              0         0        0.00     -1          1
## 3              4              5         4        1.65      1          0
## 4              6              7         1        7.05      1          0
## 5              0              0         0        0.00     -1          3
## 6              8              9         3        5.00      1          0
## 7              0              0         0        0.00     -1          3
## 8              0              0         0        0.00     -1          2
## 9             10             11         1        6.15      1          0
## 10             0              0         0        0.00     -1          2
## 11             0              0         0        0.00     -1          3
```

```r
irisP <- classCenter(training[, c(3,4)], training$Species, modFit$finalModel$prox)
irisP<- as.data.frame(irisP)
irisP$Species<- rownames(irisP)
p<- qplot(Petal.Width, Petal.Length, col=Species, data=training)
p+ geom_point(aes(x=Petal.Width, y=Petal.Length, col=Species), size=5, shape=4, data=irisP)
```

![](lecture.3_files/figure-html/unnamed-chunk-5-1.png) 

```r
pred<- predict(modFit, testing)
testing$predRight<- pred==testing$Species
table(pred, testing$Species)
```

```
##             
## pred         setosa versicolor virginica
##   setosa         15          0         0
##   versicolor      0         13         1
##   virginica       0          2        14
```

```r
qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main = "newdata Predictions")
```

![](lecture.3_files/figure-html/unnamed-chunk-5-2.png) 

##Notes

* Random forests are usually one of the two top performing algorithms along with boosting in prediction contests
* Random forests are diffictult to interpret but often very accurate
* Care should be taken to avoid overfitting (see `rfcv`) function

#Boosting

1. Take lots of (possibly) weak predictors
2. Weight them and add them up
3. Get a stronge predictor

1. Start with a set of calssifiers h1...hk
    *Examples: all possible trees, all possible regression models, all possible cutoffs
    
2. Create a classifier that combines classification functions: f(x)=sgn(sum(at*ht(x)))
    * Goal is to minimize error (on training set)
    * Iterative, select one h at each step
    * Calculate weights based on errors
    * Upweight missed classificaitons and select next h
    
[Some suggested link](http://webee.technion.ac.il/people/rmeir/BoostingTutorial.pdf)
[Adaboost](https://en.wikipedia.org/wiki/AdaBoost)

* Boosting can be used with any subset of classifiers
* One large subclass is gradient boosting
* R has multiple boosting libraries. Differences include the choice of basic classification functions and combination rules.
    * `gbm` - boosting with trees
    * `mboost` - model based boosting
    * `ada` - statistical boosting based on additive logistic regression
    * `gamBoost` - for boosting generalized additive models
* Most of these are available in the caret package


```r
library(ISLR)
data(Wage)
library(ggplot2)
library(caret)

Wage <- subset(Wage, select=-c(logwage))
inTrain<- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)

training<- Wage[inTrain,]
testing<- Wage[-inTrain,]

modFit<- train(wage ~., method="gbm", data=training, verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
## 
## Attaching package: 'plyr'
## 
## The following object is masked _by_ '.GlobalEnv':
## 
##     ozone
## 
## The following object is masked from 'package:ElemStatLearn':
## 
##     ozone
```

```r
modFit
```

```
## Stochastic Gradient Boosting 
## 
## 2102 samples
##   10 predictor
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 2102, 2102, 2102, 2102, 2102, 2102, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  RMSE      Rsquared   RMSE SD   Rsquared SD
##   1                   50      34.12745  0.3184700  1.294873  0.02437081 
##   1                  100      33.63671  0.3285735  1.267279  0.02447841 
##   1                  150      33.59487  0.3294781  1.246401  0.02407989 
##   2                   50      33.59811  0.3314369  1.289541  0.02584455 
##   2                  100      33.50160  0.3336574  1.244714  0.02555678 
##   2                  150      33.56847  0.3316295  1.208047  0.02570789 
##   3                   50      33.50423  0.3334369  1.245593  0.02559306 
##   3                  100      33.64859  0.3287606  1.163685  0.02515299 
##   3                  150      33.83100  0.3232214  1.147706  0.02488924 
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## RMSE was used to select the optimal model using  the smallest value.
## The final values used for the model were n.trees = 100,
##  interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
qplot(predict(modFit, testing), wage, data=testing)
```

![](lecture.3_files/figure-html/unnamed-chunk-6-1.png) 

* Ron Meir
* Freund and Shapire
* Slides

#Model based prediction

1. Assume that the data follow a probabilistic model
2. Use Bayes' theorem to identify optimal classifiers

##Pros

* Can take advantage of the structure of the data
* May be computationally convinient
* Are reasonably accurate on real problems

##Cons

* Make additional assumptions about the data
* When the model is incorrect you may get reduced accuracy

A range of models use this approach

* Linear discriminant analysis assumes fk(x) is multivariate Gaussian with same covariances
* Quadratic discriminant alanysys assumes fk(x) is multivariate Gaussian with different covariances
* Model based prediction assumes more complicated version for the covariance matrix
* Naive Bayes assumes independence between features for model building


```r
data(iris)
library(ggplot2)
names(iris)
```

```
## [1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width" 
## [5] "Species"
```

```r
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

```r
inTrain<- createDataPartition(y=iris$Species, p=0.7, list=FALSE)

training<- iris[inTrain,]
testing<- iris[-inTrain,]

modlda<- train(Species ~., data=training, method="lda")
```

```
## Loading required package: MASS
```

```r
modnb <- train(Species ~., data=training, method="nb")
```

```
## Loading required package: klaR
```

```r
plda <- predict(modlda, testing)
pnb<- predict(modnb, testing)
table(plda, pnb)
```

```
##             pnb
## plda         setosa versicolor virginica
##   setosa         15          0         0
##   versicolor      0         15         0
##   virginica       0          0        15
```

```r
equalPredictions <- (plda==pnb)
qplot(Petal.Width, Sepal.Width, colour=equalPredictions, data=testing)
```

![](lecture.3_files/figure-html/unnamed-chunk-7-1.png) 

Read wikipedia and the books

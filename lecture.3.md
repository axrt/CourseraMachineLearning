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
## 1) root 105 70 setosa (0.33333333 0.33333333 0.33333333)  
##   2) Petal.Length< 2.6 35  0 setosa (1.00000000 0.00000000 0.00000000) *
##   3) Petal.Length>=2.6 70 35 versicolor (0.00000000 0.50000000 0.50000000)  
##     6) Petal.Width< 1.75 36  2 versicolor (0.00000000 0.94444444 0.05555556) *
##     7) Petal.Width>=1.75 34  1 virginica (0.00000000 0.02941176 0.97058824) *
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
## [31] virginica  versicolor virginica  virginica  virginica  virginica 
## [37] virginica  versicolor virginica  virginica  virginica  versicolor
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
##   2     0.9727705  0.9587258  0.03191084   0.04814487
##   3     0.9727705  0.9587258  0.03191084   0.04814487
##   4     0.9674701  0.9504990  0.03483461   0.05269210
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
getTree(modFit$finalModel,k=2)
```

```
##    left daughter right daughter split var split point status prediction
## 1              2              3         4        0.80      1          0
## 2              0              0         0        0.00     -1          1
## 3              4              5         3        4.95      1          0
## 4              6              7         1        5.95      1          0
## 5              0              0         0        0.00     -1          3
## 6              0              0         0        0.00     -1          2
## 7              8              9         3        4.75      1          0
## 8              0              0         0        0.00     -1          2
## 9             10             11         4        1.65      1          0
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
##   versicolor      0         12         1
##   virginica       0          3        14
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
##   1                   50      35.91571  0.3136207  1.452097  0.02754667 
##   1                  100      35.31962  0.3227640  1.423374  0.02720354 
##   1                  150      35.25734  0.3232700  1.411480  0.02661931 
##   2                   50      35.29897  0.3251593  1.456077  0.02933924 
##   2                  100      35.19295  0.3255099  1.412053  0.02887486 
##   2                  150      35.27406  0.3224776  1.414362  0.02946827 
##   3                   50      35.17050  0.3276161  1.461187  0.02988218 
##   3                  100      35.27901  0.3228014  1.409451  0.03015910 
##   3                  150      35.43579  0.3176886  1.405230  0.02999237 
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## RMSE was used to select the optimal model using  the smallest value.
## The final values used for the model were n.trees = 50, interaction.depth
##  = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
qplot(predict(modFit, testing), wage, data=testing)
```

![](lecture.3_files/figure-html/unnamed-chunk-6-1.png) 

* Ron Meir
* Freund and Shapire
* Slides

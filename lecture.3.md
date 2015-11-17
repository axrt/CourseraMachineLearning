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
##     6) Petal.Length< 4.85 34  1 versicolor (0.00000000 0.97058824 0.02941176) *
##     7) Petal.Length>=4.85 36  2 virginica (0.00000000 0.05555556 0.94444444) *
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
## [19] versicolor virginica  versicolor versicolor versicolor versicolor
## [25] virginica  versicolor versicolor versicolor versicolor versicolor
## [31] virginica  virginica  virginica  versicolor virginica  virginica 
## [37] virginica  virginica  virginica  virginica  virginica  virginica 
## [43] virginica  virginica  versicolor
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

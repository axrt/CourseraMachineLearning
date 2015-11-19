# Course Project
Alexander Tuzhikov  
November 18, 2015  

#Human Activity Recognition: a Practical Machine Learning Course Project

#Syllabus

The traditional view on activity recognition has been primarily focused on which activity was performed, but not how well it was performed. A group led by Eduardo Velloso at the Lancaster University has designed [a study](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201) and generated a vast amount of data from sports gadgets such as Jawbone Up, Nike FuelBand, and Fitbit. A part of this data, made available will be studied below. Data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants will be evaluated and we will see if a good predictive model might be built, which would be able to tell if a certain exercise was performed well or inaccurately.

#Data Sources

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)


#Objectives

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how you built your model, how the cross validation was used, what we think the expected out of sample error is, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases. 

1. The submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. The text should be constrained to < 2000 words and the number of figures to be less than 5. It makes it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online.
2. Machine learning algorithm must be applied to the 20 test cases available in the test data above. The predictions should be submitted in appropriate format to the programming assignment for automated grading.

#Preparation


```r
library(dplyr)
library(caret)
set.seed(2015)
library(doParallel)
library(pander)
```

#Obtaining data


```r
#quick helper to download files
download.if.absent<- function(file, url){
        if (!file.exists(file)) {
                download.file(url, file)
        }
        return(file)
}
#download data
training.file<-"training.csv"
training.data<- read.csv(download.if.absent(file=training.file, url="http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
testing.file<-"testing.csv"
testing.data<- read.csv(download.if.absent(file=testing.file, url="http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
```

#Cleaning data


```r
#trow out values that come from the database and are irrelevant
subject.data <- training.data$user_name
training.data <- training.data[,-(1:6)]
throw.out.near.zero.variance<- function(data){
        zero.var<- nearZeroVar(data, saveMetrics=TRUE)
        data<- data[,!zero.var$nzv]
        return(data)
}
#clean training data
training.data %>% throw.out.near.zero.variance -> training.data
#check if the rest have many NAs, if so - remove those that do not have even half values
low.coverage.vals<- apply(training.data, 2, function(x){
        if(sum(is.na(x))/length(x)>=0.5){
                return(TRUE)
        }else{
                return(FALSE)
        }
})
training.data<- training.data[,!low.coverage.vals]
dim(training.data) #by now we likely have gotten rid of all useless variables
```

```
## [1] 19622    54
```

```r
#so let's get rid of the vars in training as well
#the problem_id is especially sneaky as it is not in the training dataset
testing.data[,c(setdiff(colnames(training.data), "classe"))] -> testing.data
```

#Exploratory Analysis

The remaining variable list is too large to list out, so I will go directly to PCA.


```r
training.data %>% select(-classe) %>% prcomp(center=TRUE, scale.=TRUE) -> pca.data
pca.data$x %>% as.data.frame %>% select(PC1, PC2) %>% mutate(classe=training.data$classe) %>%
        ggplot(data=., mapping=aes(x=PC1,y=PC2, color=subject.data)) +
        geom_point() + theme_bw()
```

![](project_files/figure-html/exploratory analysis-1.png) 

The plot above shows that the parameters are tightly linked to the subject the data was collected from. Interestingly, Carlitos and Eurico intersect and that makes one wonder if they attend the same gym together? =)

#Model fitting


```r
#split the data in training and testing
split<- createDataPartition(y=training.data$classe, p=0.7, list=FALSE)
mytesting.data<- training.data[-split,]
training.data<- training.data[split,]

#now proceed with model fitting
cl <- makeCluster(detectCores()/3)
registerDoParallel(cl)
#let's do cross validation
cross.val<- trainControl(method = "cv", preProcOptions = "pca", allowParallel = TRUE, number=10)
#descision tree model
rpart.model<- train(classe ~ ., method="rpart", data=training.data, trControl=cross.val)
#random forest
rf.model<- train(classe ~ ., method="rf", data=training.data, trControl=cross.val, prox=TRUE) # random forest generation here is just hellishly expensive in terms of bot cpu and ram..
#boosted model
boost.model<- train(classe ~ ., method="gbm", data=training.data, trControl=cross.val, verbose=FALSE)
#bayesian
bayesglm.model<- train(classe ~ ., method="bayesglm", data=training.data, trControl=cross.val)
stopCluster(cl)
```

Now that we have different models trained, let us see, which one is the most accurate:


```r
models<- list(Descision.Tree=rpart.model, Random.Forest=rf.model, Boosted=boost.model, Bayes.GLM=bayesglm.model)
data.frame(Model=names(models), Max.Accuracy=sapply(models, function(x){
        return(max(x$results$Accuracy))
}), Max.Kappa=sapply(models, function(x){
        return(max(x$results$Kappa))
})) %>% arrange(-Max.Accuracy, -Max.Kappa) %>% pander(caption="Accuracy and Kappa by model")
```


-----------------------------------------
    Model       Max.Accuracy   Max.Kappa 
-------------- -------------- -----------
Random.Forest      0.9973       0.9966   

   Boosted         0.9862       0.9825   

Descision.Tree     0.5526       0.4252   

  Bayes.GLM        0.4066        0.241   
-----------------------------------------

Table: Accuracy and Kappa by model

Variance estimation:


```r
confusionMatrix(predict(rf.model, mytesting.data), mytesting.data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    3    0    0    0
##          B    0 1134    2    0    0
##          C    0    2 1024    3    0
##          D    0    0    0  961    1
##          E    1    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9981   0.9969   0.9991
## Specificity            0.9993   0.9996   0.9990   0.9998   0.9998
## Pos Pred Value         0.9982   0.9982   0.9951   0.9990   0.9991
## Neg Pred Value         0.9998   0.9989   0.9996   0.9994   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1740   0.1633   0.1837
## Detection Prevalence   0.2848   0.1930   0.1749   0.1635   0.1839
## Balanced Accuracy      0.9993   0.9976   0.9985   0.9983   0.9994
```

Well, as they told us in the lecture, random forest model seems to be the most accurate. We can now predict the test set with the best two models:


```r
rf.predictions<- predict(rf.model, newdata = testing.data)
boost.predictions<- predict(boost.model, newdata=testing.data)
```

Now let's see how much the two models agree:


```r
data.frame(RandomForest=rf.predictions, Boosting=boost.predictions) %>%
        mutate(Agree=RandomForest==Boosting) %>%
        pander(caption="Number of times the best two models agree about the test set classification")
```


---------------------------------
 RandomForest   Boosting   Agree 
-------------- ---------- -------
      B            B       TRUE  

      A            A       TRUE  

      B            B       TRUE  

      A            A       TRUE  

      A            A       TRUE  

      E            E       TRUE  

      D            D       TRUE  

      B            B       TRUE  

      A            A       TRUE  

      A            A       TRUE  

      B            B       TRUE  

      C            C       TRUE  

      B            B       TRUE  

      A            A       TRUE  

      E            E       TRUE  

      E            E       TRUE  

      A            A       TRUE  

      B            B       TRUE  

      B            B       TRUE  

      B            B       TRUE  
---------------------------------

Table: Number of times the best two models agree about the test set classification

Apparently the models fully agree.

#Conclusions

Well.., apparently, random forest is the best, though the slowest model to fit. Despite that, in an independent study we have investigated and proved the proposed principle: the massive amount of data, gathered from fitness equipment, may be successfully used to analyze the quality of physical exercises performed.

#Saving Output For Submission


```r
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(rf.predictions)
```

#References
1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.

credit=read.csv("creditcard.csv")
credit=credit[,-1]
credit.label=as.matrix(credit[,30])
head(credit,100)
summary(credit)
names(credit)
table(credit$Class)
tail(credit)
library(caret)
set.seed(42)
credit.train.index=createDataPartition(credit$Class,p=0.6, list=F)
credit.train=as.data.frame(credit[credit.train.index,])
credit.test=as.data.frame(credit[-credit.train.index,])
creditM=as.matrix(credit)

###LDA
library(FactoMineR)
require(MASS)

#cross validation
credit.train.lda=lda(credit.train$Class~.,data=credit.train)
credit.train.lda.p=predict(credit.train.lda)
sum(credit.train.lda.p$class!=credit.train$Class)/length(credit.train.lda.p$class)
confusionMatrix(credit.train.lda.p$class,as.factor(credit.train$Class))
sum(credit.train.lda.p$class[credit.train$Class==1]!=credit.train$Class[credit.train$Class==1])/length(credit.train.lda.p$class[credit.train$Class==1])
#0.270
roc.curve(credit.train.lda.p$class,as.factor(credit.train$Class))
#0.923


#test
credit.test.lda.p=predict(credit.train.lda,newdata = credit.test)
sum(credit.test.lda.p$class!=credit.test$Class)/length(credit.test.lda.p$class)
#0.0005
confusionMatrix(credit.test.lda.p$class,as.factor(credit.test$Class))
roc.curve(credit.test.lda.p$class,as.factor(credit.test$Class))
#0.939
sum(credit.test.lda.p$class[credit.test$Class==1]!=credit.test$Class[credit.test$Class==1])/length(credit.test.lda.p$class[credit.test$Class==1])
#0.164

####KNN
#library(class)
#credit.knn <- knn(train=as.matrix(credit.train[,-30]),test=as.matrix(credit.test[,-30]),as.matrix(credit.train[,30]), k=5) 
#Trop lent


#arbre
library(rpart)
credit.cart=rpart(Class~., data=credit.train)
credit.cart
plot(credit.cart)

#Cross validation
credit.train.cart.prob=predict(object=credit.cart)
credit.train.cart.p=credit.train.cart.prob
credit.train.cart.p[credit.train.cart.prob>.5]=1
credit.train.cart.p[credit.train.cart.prob<=.5]=0
confusionMatrix(as.factor(credit.train.cart.p),as.factor(credit.train$Class))
sum(credit.train.cart.p!=credit.train$Class)/length(credit.train.cart.p)
#0.0005
sum(credit.train.cart.p[credit.train$Class==1]!=credit.train$Class[credit.train$Class==1])/length(credit.train.cart.p[credit.train$Class==1])
#0.249
roc.curve(as.factor(credit.train.cart.p),as.factor(credit.train$Class))
#0.971

#TEST
credit.cart.prob=predict(object=credit.cart,newdata=credit.test)
credit.cart.p=credit.cart.prob
credit.cart.p[credit.cart.prob>.5]=1
credit.cart.p[credit.cart.prob<=.5]=0
confusionMatrix(as.factor(credit.cart.p),as.factor(credit.test$Class))
mean(credit.cart.p!=credit.test$Class)
#0.0005178982
sum(credit.cart.p[credit.test$Class==1]!=credit.test$Class[credit.test$Class==1])/length(credit.cart.p[credit.test$Class==1])
#0.207
roc.curve(as.factor(credit.cart.p),as.factor(credit.test$Class))
#0.955

#Regression Logistique
library(nnet)
credit.RL <- multinom(Class~., data=credit.train)
credit.RL.p1=predict(credit.RL,credit.test)
train.label=as.factor(credit.train[,30])
test.label=as.factor(credit.test[,30])
confusionMatrix(credit.RL.p1,test.label)
sum(credit.RL.p1!=credit.test$Class)/length(credit.RL.p1)
#0.0007

sum(credit.RL.p1[credit.test$Class==1]!=credit.test$Class[credit.test$Class==1])/length(credit.RL.p1[credit.test$Class==1])
#0.318

roc.curve(credit.RL.p1,test.label)
#0.935

#Baye
library(e1071)
credit.bayes=naiveBayes(as.factor(Class)~., data=credit.train,laplace=1)
credit.bayes.p=predict(credit.bayes,credit.test)
credit.bayes.p
confusionMatrix(credit.bayes.p,test.label)
#        Reference
#Prediction      0      1
#       0    111178     23
#       1       2537    184
roc.curve(credit.bayes.p,test.label)
#0.534
sum(credit.bayes.p[test.label==1]!=test.label[test.label==1])/length(credit.bayes.p[test.label==1])
#0.111
mean(credit.bayes.p!=test.label)
#0.02

###random forest
library(randomForest)
credit.rf=randomForest(as.factor(Class)~.,data=credit.train,ntree=2000,ntry=2,maxnodes=4)
credit.rf.pred=predict(object=credit.rf,newdata=credit.test[,-30])
credit.rf.pred.r=rep(1,length(credit.test))
credit.rf.pred.r[credit.rf.pred>0.5]=1
credit.rf.pred
confusionMatrix(as.factor(credit.rf.pred),as.factor(credit.test$Class))
roc.curve(as.factor(credit.rf.pred),as.factor(credit.test$Class))
#0.931
1-mean(credit.rf.pred==credit.test$Class)
#0.0008
1-mean(credit.rf.pred[credit.test$Class==1]==credit.test$Class[credit.test$Class==1])
#0.357


 ##SVM
library(e1071)
credit.svm.poly <- svm(as.factor(Class)~., data = credit.train, kernel = "polynomial", degree=4)
credit.svm.poly.p=predict(credit.svm.poly,credit.test)
credit.svm.poly.p
confusionMatrix(as.factor(credit.svm.poly.p),as.factor(credit.test$Class))
1-mean(credit.svm.poly.p==credit.test$Class)
#0.0005
1-mean(credit.svm.poly.p[credit.test$Class==1]==credit.test$Class[credit.test$Class==1])
#0.222
roc.curve(as.factor(credit.svm.poly.p),as.factor(credit.test$Class),plotit = F)
#0.965


######Re-balanced dataset

library(ROSE)

rose=ROSE(as.factor(Class)~., data = credit, seed = 42, N=2000)$data
table(rose$Class)
table(credit$Class)
set.seed(42)
index2=createDataPartition(rose$Class,p=0.6, list=F)
rose.train=rose[index2,]
table(rose.train$Class)
set.seed(42)
rose.test=rose[-index2,]
table(rose.test$Class)

##LDA
rose.train.lda=lda(rose.train$Class~.,data=rose.train)
rose.train.lda.p=predict(rose.train.lda)
mean(rose.train.lda.p$class!=rose.train$Class)
#0.1108
confusionMatrix(rose.train.lda.p$class,as.factor(rose.train$Class))
mean(rose.train.lda.p$class[rose.test$Class==1]!=rose.train$Class[rose.test$Class==1])
#0.1350

#test
rose.test.lda.p=predict(rose.train.lda,newdata = rose.test)
mean(rose.test.lda.p$class!=rose.test$Class)
#0.14125
confusionMatrix(rose.test.lda.p$class,as.factor(rose.test$Class))

sum(rose.test.lda.p$class[rose.test$Class==1]!=rose.test$Class[rose.test$Class==1])/length(rose.test.lda.p$class[rose.test$Class==1])
#0.2857
roc.curve(rose.test.lda.p$class,as.factor(rose.test$Class))
#0.889

F=as.matrix(rose.test) %*% rose.train.lda$scaling

plot(as.matrix(rose.test[,-30]), col=as.factor(rose.test[,30]))

#KNN
library(class)
set.seed(42)
rose.knn <- knn(train=as.matrix(rose.train[,-30]),test=as.matrix(rose.test[,-30]),as.matrix(rose.train[,30]), k=1) 

table(rose.knn,rose.test$Class)
mean(rose.knn!=rose.test$Class)
#0.165
mean(rose.knn[rose.test$Class==1]!=rose.test$Class[rose.test$Class==1])
#0.335
roc.curve(rose.knn,rose.test$Class)
#0.875

rose.knn.cross <- knn(train=as.matrix(rose.train[,-30]),test=as.matrix(rose.train[,-30]),as.matrix(rose.train[,30]), k=1) 
table(rose.knn.cross,rose.train$Class)
mean(rose.knn.cross!=rose.train$Class)
#0
mean(rose.knn.cross[rose.train$Class==1]!=rose.train$Class[rose.train$Class==1])
#0

roc.curve(rose.knn.cross,rose.train$Class)
#1

###arbre de decision
library(rpart)
rose.cart=rpart(Class~., data=rose.train)
rose.cart
plot(rose.cart)
text(rose.cart)

#Cross validation
rose.train.cart.prob=predict(object=rose.cart)
rose.train.cart.p=rose.train.cart.prob
rose.train.cart.p[rose.train.cart.prob>.5]=1
rose.train.cart.p[rose.train.cart.prob<=.5]=0
confusionMatrix(as.factor(rose.train.cart.p),as.factor(rose.train$Class))
sum(rose.train.cart.p!=rose.train$Class)/length(rose.train.cart.p)
#0.02333
sum(rose.train.cart.p[rose.train$Class==1]!=rose.train$Class[rose.train$Class==1])/length(rose.train.cart.p[rose.train$Class==1])
#0.0219
roc.curve(as.factor(rose.train.cart.p),as.factor(rose.train$Class))
#0.977

#TEST
rose.cart.prob=predict(object=rose.cart,newdata=rose.test)
rose.cart.p=rose.cart.prob
rose.cart.p[rose.cart.prob>.5]=1
rose.cart.p[rose.cart.prob<=.5]=0
confusionMatrix(as.factor(rose.cart.p),as.factor(rose.test$Class))
mean(rose.cart.p!=rose.test$Class)
#0.043
sum(rose.cart.p[rose.test$Class==1]!=rose.test$Class[rose.test$Class==1])/length(rose.cart.p[rose.test$Class==1])
#0.051
roc.curve(as.factor(rose.cart.p),as.factor(rose.test$Class))
#0.956

#Regression logistique
library(nnet)
rose.RL <- multinom(Class~., data=rose.train)
rose.RL.p1=predict(rose.RL,rose.test)
train.label=as.factor(rose.train[,30])
test.label=as.factor(rose.test[,30])
confusionMatrix(rose.RL.p1,test.label)
mean(rose.RL.p1!=rose.test$Class)
#0.077
sum(rose.RL.p1[rose.test$Class==1]!=rose.test$Class[rose.test$Class==1])/length(rose.RL.p1[rose.test$Class==1])
#0.129
roc.curve(rose.RL.p1,test.label)
#0.928

#Baye
library(e1071)
rose.bayes=naiveBayes(as.factor(Class)~., data=rose.train,laplace=1)
rose.bayes.p=predict(rose.bayes,rose.test)
rose.bayes.p
confusionMatrix(rose.bayes.p,test.label)
#           Reference
#Prediction   0   1
#         0 404   0
#         1  11 385
mean(rose.bayes.p!=rose.test$Class)
#0.01375
roc.curve(rose.bayes.p,rose.test$Class)
#0.986
sum(rose.bayes.p[rose.test$Class==1]!=rose.test$Class[rose.test$Class==1])/length(rose.bayes.p[rose.test$Class==1])
#0



###random forest
library(randomForest)
rose.rf=randomForest(as.factor(Class)~.,data=rose.train,ntree=100,mtry=8,maxnodes=32)
rose.rf.pred=predict(object=rose.rf,newdata=rose.test[,-30])
rose.rf.pred.r=rep(1,length(rose.test))
rose.rf.pred.r[rose.rf.pred>0.5]=1
#rose.rf.pred
confusionMatrix(as.factor(rose.rf.pred),as.factor(rose.test$Class))
roc.curve(as.factor(rose.rf.pred),as.factor(rose.test$Class))
#0.986
1-mean(rose.rf.pred==rose.test$Class)
#0.013
1-mean(rose.rf.pred[rose.test$Class==1]==rose.test$Class[rose.test$Class==1])
#0.01298701


##SVM
library(e1071)
library(MASS)
rose.svm.poly <- svm(as.factor(Class)~., data = rose.train, kernel = "polynomial", degree=2)
rose.svm.poly.p=predict(rose.svm.poly,rose.test)
#rose.svm.poly.p
confusionMatrix(as.factor(rose.svm.poly.p),as.factor(rose.test$Class))
1-mean(rose.svm.poly.p==rose.test$Class)
#0.0675
1-mean(rose.svm.poly.p[test.label==1]==test.label[test.label==1])
#0.114
roc.curve(as.factor(rose.svm.poly.p),as.factor(rose.test$Class),plotit = F)
#0.937


rose.svm.lin <- svm(as.factor(Class)~., data = rose.train, kernel = "linear")
rose.svm.lin.p=predict(rose.svm.lin,rose.test)
#rose.svm.lin.p
confusionMatrix(as.factor(rose.svm.lin.p),as.factor(rose.test$Class))
1-mean(rose.svm.lin.p==rose.test$Class)
#0.07
1-mean(rose.svm.lin.p[test.label==1]==test.label[test.label==1])
#0.132
roc.curve(as.factor(rose.svm.lin.p),as.factor(rose.test$Class),plotit = F)
#0.937


rose.svm.sig <- svm(as.factor(Class)~., data = rose.train, kernel = "sigmoid")
rose.svm.sig.p=predict(rose.svm.sig,rose.test)
#rose.svm.sig.p
confusionMatrix(as.factor(rose.svm.sig.p),as.factor(rose.test$Class))
1-mean(rose.svm.sig.p==rose.test$Class)
#0.087
1-mean(rose.svm.sig.p[test.label==1]==test.label[test.label==1])
#0.1532
roc.curve(as.factor(rose.svm.sig.p),as.factor(rose.test$Class),plotit = F)
#0.920

rose.svm.rad <- svm(as.factor(Class)~., data = rose.train, kernel = "radial")
rose.svm.rad.p=predict(rose.svm.rad,rose.test)
#rose.svm.rad.p
confusionMatrix(as.factor(rose.svm.rad.p),as.factor(rose.test$Class))
1-mean(rose.svm.rad.p==rose.test$Class)
#0.033
1-mean(rose.svm.rad.p[test.label==1]==test.label[test.label==1])
#0.033
roc.curve(as.factor(rose.svm.rad.p),as.factor(rose.test$Class),plotit = F)
#0.966




######Variable Selection
# computing correlation of all variables with class
rose.cor <- cor(rose, as.numeric(rose$Class))
rose.cor
# ordering dimensions as a function of their correlation to the labels
rose.cor.ordered <- rose[,rev(order(abs(rose.cor)))]
rose.cor.ordered
# After some experimentation, about 75 variables seemed optimal
roseSelect <- rose.cor.ordered[, 1:6]
roseSelect
set.seed(42)
index3=createDataPartition(roseSelect$Class,p=0.6, list=F)
roseSelect.train=roseSelect[index3,]
table(roseSelect.train$Class)
set.seed(42)
roseSelect.test=roseSelect[-index2,]
table(roseSelect.test$Class)


roseSelect.train.lda=lda(roseSelect.train$Class~.,data=roseSelect.train)
roseSelect.train.lda.p=predict(roseSelect.train.lda)
mean(roseSelect.train.lda.p$class!=roseSelect.train$Class)
#0.105
confusionMatrix(roseSelect.train.lda.p$class,as.factor(roseSelect.train$Class))
mean(roseSelect.train.lda.p$class[roseSelect.test$Class==1]!=roseSelect.train$Class[roseSelect.test$Class==1])
#0.114

#test
roseSelect.test.lda.p=predict(roseSelect.train.lda,newdata = roseSelect.test)
mean(roseSelect.test.lda.p$class!=roseSelect.test$Class)
#0.118
confusionMatrix(roseSelect.test.lda.p$class,as.factor(roseSelect.test$Class))

sum(roseSelect.test.lda.p$class[roseSelect.test$Class==1]!=roseSelect.test$Class[roseSelect.test$Class==1])/length(roseSelect.test.lda.p$class[roseSelect.test$Class==1])
#0.244
#0.2857
#0.164
roc.curve(roseSelect.test.lda.p$class,as.factor(roseSelect.test$Class))
#0.906

###KNN
library(class)
set.seed(42)
roseSelect.knn <- knn(train=as.matrix(roseSelect.train[,-1]),test=as.matrix(roseSelect.test[,-1]),as.matrix(roseSelect.train[,1]), k=1) 

##test
table(roseSelect.knn,roseSelect.test$Class)
mean(roseSelect.knn!=roseSelect.test$Class)
#0.02
mean(roseSelect.knn[roseSelect.test$Class==1]!=roseSelect.test$Class[roseSelect.test$Class==1])
#0.033
roc.curve(roseSelect.knn,roseSelect.test$Class)
#0.981

roseSelect.knn.cross <- knn(train=as.matrix(roseSelect.train[,-1]),test=as.matrix(roseSelect.train[,-1]),as.matrix(roseSelect.train[,1]), k=1) 
table(roseSelect.knn.cross,roseSelect.train$Class)
mean(roseSelect.knn.cross!=roseSelect.train$Class)
#0
mean(roseSelect.knn.cross[roseSelect.train$Class==1]!=roseSelect.train$Class[roseSelect.train$Class==1])
#0

roc.curve(roseSelect.knn.cross,roseSelect.train$Class)
#1

###arbre de decision
library(rpart)
roseSelect.cart=rpart(Class~., data=roseSelect.train)
roseSelect.cart
plot(roseSelect.cart)
text(roseSelect.cart)

#Cross validation
roseSelect.train.cart.prob=predict(object=roseSelect.cart)
roseSelect.train.cart.p=roseSelect.train.cart.prob
roseSelect.train.cart.p[roseSelect.train.cart.prob>.5]=1
roseSelect.train.cart.p[roseSelect.train.cart.prob<=.5]=0
confusionMatrix(as.factor(roseSelect.train.cart.p),as.factor(roseSelect.train$Class))
sum(roseSelect.train.cart.p!=roseSelect.train$Class)/length(roseSelect.train.cart.p)
#0.0233
sum(roseSelect.train.cart.p[roseSelect.train$Class==1]!=roseSelect.train$Class[roseSelect.train$Class==1])/length(roseSelect.train.cart.p[roseSelect.train$Class==1])
#0.025
roc.curve(as.factor(roseSelect.train.cart.p),as.factor(roseSelect.train$Class))
#0.977

#TEST
roseSelect.cart.prob=predict(object=roseSelect.cart,newdata=roseSelect.test)
roseSelect.cart.p=roseSelect.cart.prob
roseSelect.cart.p[roseSelect.cart.prob>.5]=1
roseSelect.cart.p[roseSelect.cart.prob<=.5]=0
confusionMatrix(as.factor(roseSelect.cart.p),as.factor(roseSelect.test$Class))
mean(roseSelect.cart.p!=roseSelect.test$Class)
#0.02875
sum(roseSelect.cart.p[roseSelect.test$Class==1]!=roseSelect.test$Class[roseSelect.test$Class==1])/length(roseSelect.cart.p[roseSelect.test$Class==1])
#0.031
roc.curve(as.factor(roseSelect.cart.p),as.factor(roseSelect.test$Class))
#0.971


#Regression logistique
library(nnet)
roseSelect.RL <- multinom(Class~., data=roseSelect.train)
roseSelect.RL.p1=predict(roseSelect.RL,roseSelect.test)
train.label=as.factor(roseSelect.train[,1])
test.label=as.factor(roseSelect.test[,1])
confusionMatrix(roseSelect.RL.p1,test.label)
mean(roseSelect.RL.p1!=roseSelect.test$Class)
#0.072
sum(roseSelect.RL.p1[roseSelect.test$Class==1]!=roseSelect.test$Class[roseSelect.test$Class==1])/length(roseSelect.RL.p1[roseSelect.test$Class==1])
#0.135
roc.curve(roseSelect.RL.p1,roseSelect.test$Class)
#0.935

#Baye
library(e1071)
roseSelect.bayes=naiveBayes(as.factor(Class)~., data=roseSelect.train,laplace=1)
roseSelect.bayes.p=predict(roseSelect.bayes,roseSelect.test)
roseSelect.bayes.p
confusionMatrix(roseSelect.bayes.p,test.label)
#           Reference
#Prediction   0   1
#         0 404   18
#         1  5 385
mean(roseSelect.bayes.p!=roseSelect.test$Class)
#0.02875
roc.curve(roseSelect.bayes.p,test.label)
#0.972
sum(roseSelect.bayes.p[test.label==1]!=test.label[test.label==1])/length(roseSelect.bayes.p[test.label==1])
#0.04



###random forest
library(randomForest)
roseSelect.rf=randomForest(as.factor(Class)~.,data=roseSelect.train,ntree=100,mtry=5,maxnodes=32)
roseSelect.rf.pred=predict(object=roseSelect.rf,newdata=roseSelect.test[,-1])
roseSelect.rf.pred.r=rep(1,length(roseSelect.test))
roseSelect.rf.pred.r[roseSelect.rf.pred>0.5]=1
#roseSelect.rf.pred
confusionMatrix(as.factor(roseSelect.rf.pred),as.factor(roseSelect.test$Class))
roc.curve(as.factor(roseSelect.rf.pred),as.factor(roseSelect.test$Class))
#0.989
1-mean(roseSelect.rf.pred==roseSelect.test$Class)
#0.0112
mean(roseSelect.rf.pred[test.label==1]==test.label[test.label==1])
#0.984
1-mean(roseSelect.rf.pred[test.label==1]==test.label[test.label==1])
#0.015

##SVM
library(e1071)
roseSelect.svm.poly <- svm(as.factor(Class)~., data = roseSelect.train, kernel = "polynomial", degree=2)
roseSelect.svm.poly.p=predict(roseSelect.svm.poly,roseSelect.test)
#roseSelect.svm.poly.p
confusionMatrix(as.factor(roseSelect.svm.poly.p),as.factor(roseSelect.test$Class))
1-mean(roseSelect.svm.poly.p==roseSelect.test$Class)
#0.1125
1-mean(roseSelect.svm.poly.p[test.label==1]==test.label[test.label==1])
#0.215
roc.curve(as.factor(roseSelect.svm.poly.p),as.factor(roseSelect.test$Class),plotit = F)
#0.904

roseSelect.svm.lin <- svm(as.factor(Class)~., data = roseSelect.train, kernel = "linear")
roseSelect.svm.lin.p=predict(roseSelect.svm.lin,roseSelect.test)
#roseSelect.svm.lin.p
confusionMatrix(as.factor(roseSelect.svm.lin.p),as.factor(roseSelect.test$Class))
1-mean(roseSelect.svm.lin.p==roseSelect.test$Class)
#0.08
1-mean(roseSelect.svm.lin.p[test.label==1]==test.label[test.label==1])
#0.158
roc.curve(as.factor(roseSelect.svm.lin.p),as.factor(roseSelect.test$Class),plotit = F)
#0.931


roseSelect.svm.sig <- svm(as.factor(Class)~., data = roseSelect.train, kernel = "sigmoid")
roseSelect.svm.sig.p=predict(roseSelect.svm.sig,roseSelect.test)
#roseSelect.svm.sig.p
confusionMatrix(as.factor(roseSelect.svm.sig.p),as.factor(roseSelect.test$Class))
1-mean(roseSelect.svm.sig.p==roseSelect.test$Class)
#0.086
1-mean(roseSelect.svm.sig.p[test.label==1]==test.label[test.label==1])
#0.166
roc.curve(as.factor(roseSelect.svm.sig.p),as.factor(roseSelect.test$Class),plotit = F)
#0.925

roseSelect.svm.rad <- svm(as.factor(Class)~., data = roseSelect.train, kernel = "radial")
roseSelect.svm.rad.p=predict(roseSelect.svm.rad,roseSelect.test)
#roseSelect.svm.rad.p
confusionMatrix(as.factor(roseSelect.svm.rad.p),as.factor(roseSelect.test$Class))
1-mean(roseSelect.svm.rad.p==roseSelect.test$Class)
#0.028
1-mean(roseSelect.svm.rad.p[test.label==1]==test.label[test.label==1])
#0.049
roc.curve(as.factor(roseSelect.svm.rad.p),as.factor(roseSelect.test$Class),plotit = F)
#0.973

# Survivors on the Titanic - can we use a decision tree mechanism that's better than a regression?

train <- read.csv(file='train.csv',header=TRUE,sep=',')
test <- read.csv(file='test.csv',header=TRUE,sep=',')

#For regression testing
train.batch <- train[1:712, ]
validate.batch <- train[713:891, ]

#What is the proportion of people that actually survived?
#By class
mosaicplot(train$Pclass ~ train$Survived, 
           main="Passenger Fate by Traveling Class", shade=FALSE, 
           color=TRUE, xlab="Pclass", ylab="Survived")

Pclass.table <- with(train,table(Pclass,Survived))
prop.table(Pclass.table,1)

#What is the proportion of people that actually survived?
mosaicplot(train$Sex ~ train$Survived, 
           main="Passenger Fate by Gender", shade=FALSE, color=TRUE, 
           xlab="Sex", ylab="Survived")

gender.table <- with(train,table(Sex,Survived))
prop.table(gender.table,1)

mosaicplot(train$Embarked ~ train$Survived, 
           main="Passenger Fate by Port of Embarkation",
           shade=FALSE, color=TRUE, xlab="Embarked", ylab="Survived")

mosaicplot(train$AgeRange ~ train$Survived, 
           main="Passenger Fate by Age Range",
           shade=FALSE, color=TRUE, xlab="Age", ylab="Survived")

#Selected Logistic Models
model1d <- glm(formula = Survived ~ Pclass + Simplified.Salutation + AgeRange + SibSp, #Simplified.Salutation splits women by Mrs. and Ms. 'Regal' titles are place with Mrs.
               family = "binomial", data = train.batch)

summary(model1d)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.7958  -0.6149  -0.4640   0.5978   2.4922  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)                 5.4072     0.6293   8.592  < 2e-16 ***
  Pclass                     -1.0619     0.1331  -7.976 1.51e-15 ***
  Simplified.SalutationMr.   -2.5162     0.2511 -10.020  < 2e-16 ***
  Simplified.SalutationMrs.   0.5564     0.3428   1.623 0.104532    
AgeRange10-20              -1.6519     0.5314  -3.109 0.001880 ** 
  AgeRange20-30              -1.8798     0.4976  -3.777 0.000159 ***
  AgeRange30-40              -1.5801     0.5131  -3.079 0.002074 ** 
  AgeRange40-50              -2.3900     0.5657  -4.225 2.39e-05 ***
  AgeRange50-60              -2.7382     0.6442  -4.251 2.13e-05 ***
  AgeRange60+                -2.8120     0.7238  -3.885 0.000102 ***
  AgeRangeUnknown            -1.8503     0.5040  -3.671 0.000241 ***
  SibSp                      -0.4572     0.1342  -3.408 0.000655 ***
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


glm.pred <- predict(model1d, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1
confusionMatrix(glm.pred, validate.batch$Survived)

         Reference
Prediction   0   1
         0 108  21
         1   7  43
Accuracy : 0.8436

model1f <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp, #Sex is only male and female
               family = "binomial", data = train.batch)
summary(model1f)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.8305  -0.6219  -0.4612   0.6019   2.4911  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)       5.4889     0.6287   8.731  < 2e-16 ***
  Pclass           -1.0656     0.1326  -8.037 9.19e-16 ***
  Sexmale          -2.7364     0.2171 -12.602  < 2e-16 ***
  AgeRange10-20    -1.5988     0.5335  -2.997 0.002729 ** 
  AgeRange20-30    -1.7336     0.4913  -3.529 0.000418 ***
  AgeRange30-40    -1.4117     0.5033  -2.805 0.005035 ** 
  AgeRange40-50    -2.1661     0.5464  -3.964 7.37e-05 ***
  AgeRange50-60    -2.5758     0.6342  -4.061 4.88e-05 ***
  AgeRange60+      -2.6466     0.7150  -3.701 0.000214 ***
  AgeRangeUnknown  -1.7411     0.5007  -3.478 0.000506 ***
  SibSp            -0.4357     0.1316  -3.311 0.000929 ***

glm.pred <- predict(model1f, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1

confusionMatrix(glm.pred, validate.batch$Survived)

Confusion Matrix and Statistics

         Reference
Prediction   0   1
         0 106  22
         1   9  42

Accuracy : 0.8268  

model1g <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp + Regal + Authority, #Regal is a binary if the title is such, Authority is a binary if the title is such.
               family = "binomial", data = train.batch)
summary(model1g)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.8234  -0.6164  -0.4592   0.6038   2.4928  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)       5.4615     0.6308   8.658  < 2e-16 ***
  Pclass           -1.0496     0.1331  -7.888 3.07e-15 ***
  Sexmale          -2.7298     0.2182 -12.511  < 2e-16 ***
  AgeRange10-20    -1.6089     0.5340  -3.013 0.002588 ** 
  AgeRange20-30    -1.7520     0.4918  -3.563 0.000367 ***
  AgeRange30-40    -1.4187     0.5038  -2.816 0.004859 ** 
  AgeRange40-50    -2.2322     0.5522  -4.042 5.29e-05 ***
  AgeRange50-60    -2.5352     0.6409  -3.955 7.64e-05 ***
  AgeRange60+      -2.6199     0.7168  -3.655 0.000257 ***
  AgeRangeUnknown  -1.7544     0.5013  -3.499 0.000466 ***
  SibSp            -0.4449     0.1324  -3.362 0.000775 ***
  Regal            14.7431   548.5739   0.027 0.978559    
Authority        -0.4833     1.1336  -0.426 0.669890   

glm.pred <- predict(model1g, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1

confusionMatrix(glm.pred, validate.batch$Survived)

         Reference
Prediction   0   1
         0 106  22
         1   9  42

Accuracy : 0.8268     

model1h <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp + Length, #Sex is only male and female
               family = "binomial", data = train.batch)
summary(model1h)

glm.pred <- predict(model1h, test)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1
confusionMatrix(glm.pred, validate.batch$Survived)

logisticmodel2 <- cbind(test$PassengerId,glm.pred)
colnames(logisticmodel2) <- c("PassengerId", "Survived")
write.csv(logisticmodel2, file = "logisticmodel2.csv", row.names = FALSE)

model1i <- glm(formula = Survived ~ I(Pclass^2) + Sex + AgeRange + SibSp + Length, #Sex is only male and female
               family = "binomial", data = train.batch)
summary(model1i)

model1i <- glm(formula = Survived ~ I(Pclass^2) + Sex + AgeRange + SibSp + Length, #Sex is only male and female
               family = "binomial", data = validate.batch)
glm.pred <- predict(model1i, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1
confusionMatrix(glm.pred, validate.batch$Survived)



model1j <- glm(formula = Survived ~ I(Pclass^2) + Sex + MiddleAged + UnknownAge + SibSp + Length,
               family = "binomial", data = train.batch)
summary(model1j)

model1i <- glm(formula = Survived ~ I(Pclass^2) + Sex + AgeRange + SibSp + Length, #Sex is only male and female
               family = "binomial", data = validate.batch)
glm.pred <- predict(model1i, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1
confusionMatrix(glm.pred, validate.batch$Survived)

#Last Name ends in F or C
model1k <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp + Length + EndsinF + EndsinC,
               family = "binomial", data = train.batch)
summary(model1k)

model1k <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp + Length + EndsinF + EndsinC,
               family = "binomial", data = validate.batch)
glm.pred <- predict(model1k, validate.batch)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1
confusionMatrix(glm.pred, validate.batch$Survived)

logisticmodel2 <- cbind(test$PassengerId,glm.pred)
colnames(logisticmodel2) <- c("PassengerId", "Survived")
write.csv(logisticmodel2, file = "logisticmodel2.csv", row.names = FALSE)

###So which logicistic model do we choose?
glm.pred <- predict(model1f, test)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1

logisticmodel <- cbind(test$PassengerId,glm.pred)
colnames(logisticmodel) <- c("PassengerId", "Survived")
write.csv(logisticmodel, file = "logisticmodel.csv", row.names = FALSE)

The result? #1210  new	 Harrison Adler	#0.76555 #2	 Mon, 16 Jun 2014 01:02:21

#Tied with 225 other people! We can do better!

glm.pred <- predict(model1k, test)
glm.pred[glm.pred < .5] = 0
glm.pred[glm.pred > .5] = 1

logisticmodel2 <- cbind(test$PassengerId,glm.pred)
colnames(logisticmodel2) <- c("PassengerId", "Survived")
write.csv(logisticmodel2, file = "logisticmodel2.csv", row.names = FALSE)

#Decision Trees 
library(rpart)
model2 <- rpart(Survived~Pclass + Sex + AgeRange + SibSp + Regal + Authority, data=train,control=rpart.control(minbucket=1)) #Important to specify as Survival is a factor
model2 <- rpart(as.factor(Survived)~Pclass + Sex + AgeRange + SibSp + Regal + Authority, data=train,control=rpart.control(minbucket=1))
library(rattle)
fancyRpartPlot(model2)
decisiontreemodel <- predict(model2,newdata=test)
survivalprobability <- decisiontreemodel[,2]
survivalprobability <- round(survivalprobability,0)
decisiontree <- cbind(test$PassengerId,survivalprobability)
colnames(decisiontree) <- c("PassengerId", "Survived")
write.csv(decisiontree, file = "decisiontree.csv", row.names = FALSE)

#This moved me up one notch, but still tied with 56 other people

library(rpart)
model3 <- rpart(as.factor(Survived)~Pclass + Sex + AgeRange + SibSp + Length + EndsinF + EndsinC, data=train,control=rpart.control(minbucket=1)) #Important to specify as factor
library(rattle)
fancyRpartPlot(model3)
decisiontreemodel <- predict(model3,newdata=test)
survivalprobability <- decisiontreemodel[,2]
survivalprobability <- round(survivalprobability,0)
decisiontree3 <- cbind(test$PassengerId,survivalprobability)
colnames(decisiontree3) <- c("PassengerId", "Survived")
write.csv(decisiontree3, file = "decisiontree2.csv", row.names = FALSE)

#Cluster Method (How far away are points from the mean and SD)
train <- read.csv(file='train2.csv',header=TRUE,sep=',')
test <- read.csv(file='test2.csv',header=TRUE,sep=',')
means <- sapply(test, mean)
SD <- sapply(test, sd)
dataScale <- scale(test, center = means, scale = SD)
Dist <- dist(dataScale, method = "euclidean")
clusterModel <- hclust(Dist, method = "ward")
plot(clusterModel)

result <- cutree(clusterModel, k = 2)
result <- result - 1
count(result)
cluster <- cbind(test$PassengerId,result)
colnames(cluster) <- c("PassengerId", "Survived")
write.csv(cluster, file = "cluster.csv", row.names = FALSE)

#Bagging
library(rpart)
train <- read.csv(file='train2.csv',header=TRUE,sep=',')
test <- read.csv(file='test2.csv',header=TRUE,sep=',')

pre.matrix <- matrix(0,30,418)
for(i in 1:30){
  sample_index <- sample(1:10,10,replace=T)
  sample_data <- train[sample_index,]
  stump <- rpart(Survived~.,data=train,
                 method='class',
                 control=rpart.control(minbucket=1,maxdepth=1)) # stump learner, a weaker learner
  pre <- predict(stump,newdata=test,type='class')
  pre.matrix[i,] <- as.numeric(as.character(pre))
}
vote_pre <- apply(pre.matrix,2,sum)
final_pre <- sign(vote_pre)

bagging <- cbind(test$PassengerId,final_pre)
colnames(bagging) <- c("PassengerId", "Survived")
write.csv(bagging, file = "bagging.csv", row.names = FALSE)
#Still no improvement, only 77% accurate

#Random Forest - Needs Debugging
library(randomForest)
rf <- randomForest(Survived ~ ., x = train[,3:21],y=train[,-2],xtest=test[,2:20])
rf

library(caret)
ctrl <- trainControl(method = "cv", number = 10)

auto-tune a random forest
grid_rf <- data.frame(.mtry = c(2, 4, 8, 16))

set.seed(300)
m_rf <- train(Survived ~ ., data = credit, method = "rf",
              trControl = ctrl,
              tuneGrid = grid_rf)

# Survivors on the Titanic - can we use a decision tree mechanism that's better than a regression?

train <- read.csv(file='train.csv',header=TRUE,sep=',')
test <- read.csv(file='test.csv',header=TRUE,sep=',')

missing.types <- c("NA", "")
train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
test.column.types <- train.column.types[-2]

train.batch <- train[1:712, ]
test.batch <- train[713:891, ]

#Passenger ID, for internal reference
#Survived - this is what we are trying to identify
#Pclass - Passenger Class, very useful
#Name, safe to ignore at the moment
#Sex, likely useful
#Age - likely useful, can an NA indicate a non-survivor?
#SibSp - Number of Siblings/Spouses Aboard
#Parch - Number of Parents/Children Aboard
#Ticket - ignore at the moment
#Fare - include
#Cabin - ignore at the moment
#Embarked - include, don't forget 2 with blank data

#What is the proportion of people that actually survived?
gender.table <- with(train,table(Sex,Survived))
prop.table(gender.table,1)

Survived
Sex              0         1
female 0.2579618 0.7420382
male   0.8110919 0.1889081

#By observation, we know that gender is going to be a good predictor. What about salutation?

# Salutation cleanup
train$Title = NA
titles  = c("Master","Miss","Mrs","Mr","Dr","Col","Major","Capt","Rev","Don","Sir","Lady","Ms","Mme","Mlle","Countess","Jonkheer")
titles_ = paste0(titles,".")
for (i in 1:length(titles)){
  train$Title[grep(titles_[i],train$Name, fixed=TRUE)] = titles[i]
}
train$Title = factor(train$Title)

#Salutation Simplified into Mr, Miss, and Mrs.
#Assume that all females with an unusual title are Mrs.
train$TitleSimplified = NA
train$TitleSimplified <- as.matrix(train$TitleSimplified)
for (i in 1:length(train)) {
  if(train$Sex[i]=='female') {
    if(train$Title[i]=='Miss') {
      train$TitleSimplified[i]=='Miss'
    } else {
      train$TitleSimplified[i]=='Mrs'
    }
  }
   else {
     train$TitleSimplified[i]=='Mr'
  }
}

train$Title = NA
titles  = c("Master","Miss","Mrs","Mr","Dr","Col","Major","Capt","Rev","Don","Sir","Lady","Ms","Mme","Mlle","Countess","Jonkheer")
titles_ = paste0(titles,".")
for (i in 1:length(titles)){
  train$Title[grep(titles_[i],train$Name, fixed=TRUE)] = titles[i]
}
train$Title = factor(train$Title)

salutation.table <- with(train,table(Title,Survived))
prop.table(salutation.table,1)

Survived
Salutation               0         1
Capt.          1.0000000 0.0000000
Col.           0.5000000 0.5000000
Don.           1.0000000 0.0000000
Dr.            0.5714286 0.4285714
Jonkheer.      1.0000000 0.0000000
Lady.          0.0000000 1.0000000
Major.         0.5000000 0.5000000
Master.        0.4250000 0.5750000
Miss.          0.3021978 0.6978022
Mlle.          0.0000000 1.0000000
Mme.           0.0000000 1.0000000
Mr.            0.8433269 0.1566731
Mrs.           0.2080000 0.7920000
Ms.            0.0000000 1.0000000
Rev.           1.0000000 0.0000000
Sir.           0.0000000 1.0000000
the Countess.  0.0000000 1.0000000

#So we see that 69.8% of women with the 'Miss' salutation survived, compared to 79.2% with 'Mrs.'

#Checking for outliers
library(mice)
md.pattern(train)
md.pattern(test)
flux(train)
pobs    influx outflux      ainb       aout      fico
PassengerId 1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Survived    1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Pclass      1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Name        1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Salutation  1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
First.Name  1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Last.Name   1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Sex         1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Age         0.8013468 0.1878981       0 0.9333333 0.00000000 0.0000000 #18.8% of the data is missing
SibSp       1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Parch       1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Ticket      1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Fare        1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Cabin       1.0000000 0.0000000       1       NaN 0.01324355 0.1986532
Embarked    1.0000000 0.0000000       1       NaN 0.01324355 0.1986532

#Age is the signifcant variable with NAs - no age, you're less important
#To deal with this, I will bucket age groups, and inlcude a special bucket for unknown age

train

#What can we visualize?
GetSurvivalProb = function(Survival,categories){
  counts2byN <- table(train$Survived, categories)
  return(counts2byN[2,]/apply(counts2byN,2,sum))
}

GetSurvivalProb(train$Survived, train$Sex)
GetSurvivalProb(train$Survived, train$Pclass)
GetSurvivalProb(train$Survived, train$SibSp)
GetSurvivalProb(train$Survived, train$Parch)
GetSurvivalProb(train$Survived, train$Embarked)
plot(GetSurvivalProb(train$Survived, train$Age)) #let's fix this and put into bins

#First, let's try a logistic regression.
#http://nycdatascience.com/slides/RIntermediate/part2/index.html#101
#May look scary, but we're simply saying we have a binary variable of which there is probability p of occurring an 1-p of not occurring.
model1 <- glm(Survived~Pclass+Title+Age+SibSp+Parch,data=train,family='binomial')
summary(model1)

all:
  glm(formula = Survived ~ Pclass + Title + Age + SibSp + Parch, 
      family = "binomial", data = train)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.4331  -0.5759  -0.3339   0.5188   2.4668  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)   -1.198e+01  2.400e+03  -0.005 0.996017    
Pclass        -1.324e+00  1.501e-01  -8.822  < 2e-16 ***
  TitleCol       1.531e+01  2.400e+03   0.006 0.994909    
TitleCountess  3.101e+01  3.393e+03   0.009 0.992709    
TitleDon      -1.879e+00  3.393e+03  -0.001 0.999558    
TitleDr        1.560e+01  2.400e+03   0.007 0.994811    
TitleJonkheer -1.948e+00  3.393e+03  -0.001 0.999542    
TitleLady      3.212e+01  3.393e+03   0.009 0.992447    
TitleMajor     1.498e+01  2.400e+03   0.006 0.995019    
TitleMaster    1.779e+01  2.400e+03   0.007 0.994085    
TitleMiss      1.731e+01  2.400e+03   0.007 0.994243    
TitleMlle      3.070e+01  2.939e+03   0.010 0.991665    
TitleMme       3.070e+01  3.393e+03   0.009 0.992782    
TitleMr        1.451e+01  2.400e+03   0.006 0.995174    
TitleMrs       1.816e+01  2.400e+03   0.008 0.993961    
TitleMs        3.216e+01  3.393e+03   0.009 0.992438    
TitleRev      -3.772e-01  2.583e+03   0.000 0.999883    
TitleSir       3.216e+01  3.393e+03   0.009 0.992439    
Age           -3.459e-02  9.775e-03  -3.539 0.000402 ***
  SibSp         -5.949e-01  1.359e-01  -4.378  1.2e-05 ***
  Parch         -2.464e-01  1.343e-01  -1.835 0.066538 .  
---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 964.52  on 713  degrees of freedom
Residual deviance: 583.36  on 693  degrees of freedom
(177 observations deleted due to missingness)
AIC: 625.36

Number of Fisher Scoring iterations: 15

#Title does not seem the best way to go forth here

model1a <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch, 
    family = "binomial", data = train)

summary(model1a)

model1b <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp, 
               family = "binomial", data = train)

summary(model1b)

model1c <- glm(formula = Survived ~ Pclass + Sex + AgeRange + SibSp, 
               family = "binomial", data = train)

summary(model1c)

model1d <- glm(formula = Survived ~ Pclass + Simplified.Salutation + AgeRange + SibSp, 
               family = "binomial", data = train)

summary(model1d)

#This is the final model I'm sticking with

Call:
  glm(formula = Survived ~ Pclass + Simplified.Salutation + AgeRange + 
        SibSp, family = "binomial", data = train)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.9260  -0.5996  -0.4501   0.5930   2.5766  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)                 5.9077     0.5792  10.200  < 2e-16 ***
  Pclass                     -1.1187     0.1208  -9.262  < 2e-16 ***
  Simplified.SalutationMr.   -2.5064     0.2302 -10.889  < 2e-16 ***
  Simplified.SalutationMrs.   0.7349     0.3125   2.352   0.0187 *  
  AgeRange10-20              -2.0402     0.4736  -4.308 1.65e-05 ***
  AgeRange20-30              -2.2667     0.4416  -5.133 2.85e-07 ***
  AgeRange30-40              -2.1833     0.4565  -4.782 1.73e-06 ***
  AgeRange40-50              -2.8522     0.5121  -5.569 2.56e-08 ***
  AgeRange50-60              -3.1205     0.5841  -5.342 9.19e-08 ***
  AgeRange60+                -3.2869     0.6772  -4.853 1.21e-06 ***
  AgeRangeUnknown            -2.2837     0.4514  -5.059 4.21e-07 ***
  SibSp                      -0.5221     0.1251  -4.172 3.02e-05 ***
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 1186.66  on 890  degrees of freedom
Residual deviance:  772.05  on 879  degrees of freedom
AIC: 796.05

Number of Fisher Scoring iterations: 5

train.glm <- glm(formula = Survived ~ Pclass + Simplified.Salutation + AgeRange + SibSp, family = "binomial", data = train)
p.glm = predict(model1d,data=test,type="response")

confusionMatrix(p.glm, test$Survived)

p.glm01 <- p.glm
p.glm01[p.glm < .5] = 0
p.glm01[p.glm > .5] = 1

logisticmodel1.sub <- cbind(test$PassengerId,p.glm01)
colnames(logisticmodel1.sub) <- c("PassengerId", "Survived")
write.csv(logisticmodel1.sub, file = "logisticmodel1.csv", row.names = FALSE)

#So let's get into the decision tree!
#http://nycdatascience.com/slides/RIntermediate/part4/index.html#5
library(rpart)
model2 <- rpart(Survived~Pclass + Title + AgeRange + SibSp + Parch, data=train,control=rpart.control(minbucket=1))
model2 <- rpart(as.factor(Survived)~Pclass + Simplified.Salutation + AgeRange + SibSp, data=train,control=rpart.control(minbucket=1))
library(rattle)
fancyRpartPlot(model2)

#What does this plot tell us?
#If your title is Capt, Doc, Dr, Jonkheer, Mr or Rev, there's an 84% probability you wound up dead on the Titanic.
#Female (or other title not above) passengers in 1st or 2nd class had a 94% probability you survived. Whew!
#3rd class female (or other title not above) passengers who boarded with 3 or more siblings (or 2 siblings + spouse) were not in luck - 89% probability of perish.
#3rd class female (or other title not above) passengers who boarded with 2 or fewer siblings (or 1 sibling + spouse) found their age to be vital. 72% of those middle aged (20-50) perished, while 62% of those outside of that age bracket survived.

#This is really good! Can we make it better?

#Naive Bayes
library(e1071)
library(plyr)
data <- data.frame(train$Survived,train$Pclass,train$Salutation.Key,train$Age.Range.Key,train$SibSp)
ddply(data,train$Survived,function(x)colwise(mean)(x))
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data_x <- as.data.frame(lapply(data[,-1], normalize))
data_x <- t(apply(data_x,1,factor))
model <- naiveBayes(data_x, data[,1],laplace=1)
pre <- predict(model,newdata=data_x)
table(pre,levels(data[,1])) #Still needs work

#K Nearest Neighbors
library(caret)
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10,
                           repeats = 3)

knnModel <- train(y=data[,1],
                  x=data[,-1],
                  method = "knn",
                  preProc = c("center", "scale"),
                  tuneGrid = data.frame(.k = 1:20),
                  trControl = fitControl)
plot(knnModel) #This still needs work too

library(class)
# cross-validation
knn_cv <- function(x,y,n=5,k){
  m <- nrow(x)
  num <- sample(1:n,m,replace=T)
  res <- numeric(n)
  for (i in 1:n) {
    x.t <- x[num!=i, ]
    x.v <- x[num==i, ]
    y.t <- y[num!=i]
    y.v <- y[num==i]
    pred <- knn(train=x.t,test=x.v,cl=y.t,k=k)
    accu <- sum(pred==y.v)/length(pred)
    res[i] <- accu
  }
  return(mean(res))
}

#RandomForest
Time to give the popular Random Forest (RF) model a shot at the Titanic challenge. The number of randomly pre-selected predictor variables for each node, designated mtry, is the sole parameter available for tuning an RF with train. Since the number of features is so small, there really isn't much scope for tuning mtry in this case. Nevertheless, I'll demonstrate here how it can be done. Let's have mtry=2 and mtry=3 duke it out over the Titanic data.

rf.grid <- data.frame(.mtry = c(2, 3))
set.seed(35)
rf.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
data = train.batch,
method = "rf",
metric = "ROC",
tuneGrid = rf.grid,
trControl = cv.ctrl)


#What is the best model?
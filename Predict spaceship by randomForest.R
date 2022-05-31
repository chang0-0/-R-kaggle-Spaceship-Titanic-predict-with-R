library(tidyr)
library(scales)
library(Hmisc)
library(ggplot2)
library(dplyr)
library(randomForest)
library(caret)
library(tidyverse)

setwd("C:/Users/Samsung/Desktop/빅분기실기준비/우주선타이타닉 예측")
list.files()


# test data have not 'Transported' column
train = read.csv(
    file = 'train.csv',
    encoding = 'UTF-8',
    na.strings = c("", "na", "NA"),  
    stringsAsFactor = TRUE
)

test = read.csv(
    file = 'test.csv',
    encoding = 'UTF-8',
    na.strings = c("", "na", "NA"),
    stringsAsFactor = TRUE
)

nrow(train)
str(train)
str(test)
summary(train)
total <- nrow(train) + nrow(test)
nrow(test)
scales::percent(nrow(train) / total) # "67%"
scales::percent(nrow(test) / total) # "33%"
# 전체 데이터의 67%는 train 데이터, 33%는 test 데이터


# all데이터 하나로 통합
test$Transported <- NA
all <- rbind(train, test)
str(all)
all <- all %>% select(-Name, -PassengerId)

# 결측치 제거 및 형변환
all$CryoSleep <- as.factor(all$CryoSleep)
all$Cabin <- as.character(all$Cabin)
str(all)

sapply(all, function(x) {
    sum(is.na(x))
})


all$Age <- impute(all$Age, mean)
all$CryoSleep <- impute(all$CryoSleep, mean)
all$VRDeck <- impute(all$VRDeck, mean)
all$ShoppingMall <- impute(all$ShoppingMall, mean)
all$RoomService <- impute(all$RoomService, mean)
all$FoodCourt <- impute(all$FoodCourt, mean)
all$VIP <- impute(all$FoodCourt, median)
all$Spa <- impute(all$Spa, mean)
all$HomePlanet <- impute(all$HomePlanet, mean)
all <- all%>%fill(Destination,Cabin,HomePlanet,CryoSleep, .direction="updown")

test <- all %>% filter( is.na(Transported)  )
train <- all %>% filter( !is.na(Transported)  )
nrow(train)
nrow(test)




train$Transported <- ifelse(
    train$Transported == 'TRUE',
    train$Transported <- 1,
    train$Transported <- 0
)

train$Transported <- as.factor(as.character(train$Transported))
table(train$Transported )

md_rf <- randomForest(
    Transported ~ .,
    train,
    do.trace = TRUE,
    ntree = 500
)

md_rf

md_rf.pred <- predict(
    md_rf,
    newdata = test
)

head(md_rf.pred)

md_rf.pred <- ifelse(
    md_rf.pred >= 0.5,
    md_rf.pred <- "True",
    md_rf.pred <- "False"
)

md_rf.pred <- as.factor(md_rf.pred)

levels(train$Transported)
levels(md_rf.pred)
head(md_rf.pred)

md_rf.pred <- as.factor(md_rf.pred)
head(md_rf.pred)


str(valid)

valid <- valid$Transported
head(valid)

valid <- as.factor(valid)
levels(valid) <- c("True", "False")
head(valid)


str(train)

confusionMatrix(
    data = md_rf.pred,
    refer = train$Transported,
    positive = "True"
)

valid <- read.csv("sample_submission.csv")
result <- data.frame(
    valid$PassengerId,
    md_rf.pred
)

# 결과 데이터 생성 및 검증
names(result) <- c("PassengerId", "Transported")
write.csv(result, "randomForest_predict.csv", row.names = FALSE)
result2 <- read.csv("result.csv")
head(result2)

---
title: "Campaign Conversion Forecasting"
author: "Yue CHEN"
date: "2021年11月27日"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r message=TRUE, include=FALSE}
library(RODBC)
library(glmnet)
library(dplyr)
library(recipes)
library(caret)
library(xgboost)
# library(sqldf)
# library(tidyverse)
```

## Read Data
```{r}
# Connect to data source
db = odbcConnect("localhost_3306", uid="root", pwd="xxxxxxxxx")
sqlQuery(db, "USE ma_charity_full")

# query for features and if donor/donor_amount
query = "
select 
    ass.contact_id,
    recency,
    frequency,
		recent_frequency,
    avgamount,
    maxamount,
    calibration,
    donation,
    amount
from 
    assignment2 ass
    left join 
        (select 
            contact_id,
            DATEDIFF(20180626, MAX(a.act_date)) / 365 AS recency,
            COUNT(a.amount) AS frequency,
						0.000000000001 + SUM(case when act_date >= 20150626 then 1 else 0 end)*1.00/ DATEDIFF(MAX(a.act_date), 20150626) *365 AS recent_frequency,
            AVG(a.amount) AS avgamount,
            MAX(a.amount) AS maxamount
        from 
            acts a
        where 
            (act_date <  20180626) and
            (act_type_id = 'DO')
        group by 
            contact_id
        ) ac on ass.contact_id = ac.contact_id"

df_full = sqlQuery(db, query)
tail(df_full)

# Close the connection
odbcClose(db)
```

# Split Dataset
```{r }
# prepare dataset
df_dona = df_full[!is.na(df_full$donation),]

# SPLIT test set
df_dona_0 <- df_dona[which(df_dona$donation==0),]
df_dona_1 <- df_dona[which(df_dona$donation==1),]
split <- 0.7
n <- as.integer(dim(df_dona_0)[1]*split)
m <- as.integer(dim(df_dona_1)[1]*split)

# TRAINING set
train_index_0 <- sample(1:nrow(df_dona_0), n, replace=F)
train_index_1 <- sample(1:nrow(df_dona_1), m, replace=F)
train_0 <- df_dona_0[train_index_0,]
train_1 <- df_dona_1[train_index_1,]
train_dona <- rbind(train_0, train_1)

# TEST set
test_0 <- df_dona_0[-train_index_0, ]
test_1 <- df_dona_1[-train_index_1, ]
test_dona <- rbind(test_0, test_1)

# set for pred finnally
df_4_pred = df_full[is.na(df_full$donation),]
```

# BUILDING XGBOOST DONATION MODEL
## GET Parameter
```{r}
# Data cleaning
# remove NA from training set
train_dona_no_amount = train_dona[!is.na(train_dona$frequency) & !is.na(train_dona$recent_frequency), ]

# for donations
x = cbind(train_dona_no_amount$recency, 
          train_dona_no_amount$frequency, 
          train_dona_no_amount$recent_frequency,
          log(train_dona_no_amount$recency), 
          log(train_dona_no_amount$frequency), 
          log(train_dona_no_amount$recent_frequency), 
          train_dona_no_amount$recency * train_dona_no_amount$frequency)
colnames(x) = 1:7
y = as.matrix(train_dona_no_amount$donation)
dtrain = xgb.DMatrix(data=x,label = y)

# set parameter for iteration
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

# grid search/cross validation for best parameter
for (iter in 1:50) {
    param <- list(objective = "binary:logistic",     #目标函数：logistic的二分类模型，因为Y值是二元的
          eval_metric = c("logloss"),                # 评估指标：logloss
          max_depth = sample(3:8, 1),               # 最大深度的调节范围：1个 6-10 区间的数
          eta = runif(1, .01, .3),                   # eta收缩步长调节范围：1个 0.01-0.3区间的数
          gamma = runif(1, 0.0, 0.2),                # gamma最小损失调节范围：1个 0-0.2区间的数
          subsample = runif(1, .6, .9),             
          colsample_bytree = 1, 
          min_child_weight = sample(1:10, 1),
          max_delta_step = sample(1:10, 1)
          )
    cv.nround = 50                                   # 迭代次数：50
    cv.nfold = 5                                     # 5折交叉验证
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, metrics=c("auc","rmse","error"),
                    nfold=cv.nfold, nrounds=cv.nround, watchlist = list(),
                    verbose = F, early_stop_round=8, maximize=FALSE)

    min_logloss = min(mdcv$evaluation_log[,test_logloss_mean])
    min_logloss_index = which.min(mdcv$evaluation_log[,test_logloss_mean])

    if (min_logloss < best_logloss) {
        best_logloss = min_logloss
        best_logloss_index = min_logloss_index
        best_seednumber = seed.number
        best_param = param
    }
}
```

## TRAIN XGBOOST MODEL
```{r}
# get parameter for xgboost
(nround = best_logloss_index)
set.seed(best_seednumber)
best_seednumber
(best_param)                

xgb_model <- xgb.train(data=dtrain, params=best_param, nrounds=nround, nthread=6, watchlist = list())

# Prepare new data as matrix
newdata2 = cbind(test_dona$recency, 
                 test_dona$frequency, 
                 test_dona$recent_frequency,
                 log(test_dona$recency), 
                 log(test_dona$frequency), 
                 log(test_dona$recent_frequency),
                 test_dona$recency * test_dona$frequency)
# make predictions by specifying a specific value for lambda
probs = predict(xgb_model, newdata2, type = "response")
# gain chart explore
  # Rank order target variable in decreasing order of (predicted) probability
  target = test_dona$donation[order(probs, decreasing=TRUE)] / sum(test_dona$donation)
  gainchart = c(0, cumsum(target))
  # Create a random selection sequence
  random = seq(0, to = 1, length.out = length(test_dona$donation))
  # Create the "perfect" selection sequence
  perfect = test_dona$donation[order(test_dona$donation, decreasing=TRUE)] / sum(test_dona$donation)
  perfect = c(0, cumsum(perfect))
  # Plot gain chart, add random line
  plot(gainchart)
  lines(random)
  lines(perfect)
```

## Apply to the df_4_pred
```{r}
newdata2 = cbind(df_4_pred$recency, 
                 df_4_pred$frequency, 
                 df_4_pred$recent_frequency,
                 log(df_4_pred$recency), 
                 log(df_4_pred$frequency),
                 log(df_4_pred$recent_frequency),
                 df_4_pred$recency * df_4_pred$frequency)
colnames(newdata2) = 1:7
if_dona = predict(xgb_model, newdata2, type = "response")
summary(if_dona)
df_4_pred$donation <- if_dona
```

# BUILDING DONATION AMOUNT MODEL
```{r }
# lasso model
# filter traing data with na feature 
train_dona_am = train_1[!is.na(train_1$recency), ]
# set x,y
x = cbind(train_dona_am$recency, 
          train_dona_am$frequency, 
          train_dona_am$recent_frequency,
          train_dona_am$avgamount,
          train_dona_am$maxamount,
          log(train_dona_am$recency), 
          log(train_dona_am$frequency),
          log(train_dona_am$recent_frequency),
          log(train_dona_am$avgamount),
          log(train_dona_am$maxamount),
          train_dona_am$recency * train_dona_am$frequency)
y = train_dona_am$amount
# Complete model with lasso penalty
# Step 1: fit the model for various values of lambda
lasso_4_amount = glmnet(x, y, family = "gaussian", alpha=1)
plot(lasso_4_amount, xvar="lambda")
# Step 2: find the optimal value of lambda using cross-validation
cv.lasso_4_amount = cv.glmnet(x, y, family = "gaussian", alpha=1)
plot(cv.lasso_4_amount)
best.lambda = cv.lasso_4_amount$lambda.min
# Prepare new data as matrix
newdata2 = cbind(test_1$recency, 
          test_1$frequency,
          test_1$recent_frequency,
          test_1$avgamount,
          test_1$maxamount,
          log(test_1$recency), 
          log(test_1$frequency),
          log(test_1$recent_frequency),
          log(test_1$avgamount),
          log(test_1$maxamount),
          test_1$recency * test_1$frequency)
# Step 3: make predictions by specifying a specific value for lambda
amount = predict(lasso_4_amount, newdata2, s = best.lambda)
# Step 4： gain chart explore
  # Rank order target variable in decreasing order of (predicted) amount
  target = test_1$amount[order(amount, decreasing=TRUE)] / sum(test_1$amount)
  gainchart = c(0, cumsum(target))
  # Create a random selection sequence
  random = seq(0, to = 1, length.out = length(test_1$amount))
  # Create the "perfect" selection sequence
  perfect = test_1$amount[order(test_1$amount, decreasing=TRUE)] / sum(test_1$amount)
  perfect = c(0, cumsum(perfect))
  # Plot gain chart, add random line
  plot(gainchart)
  lines(random)
  lines(perfect)
```

# Apply to the df_4_pred
```{r}
# filter 
newdata2 = cbind(df_4_pred$recency, 
                  df_4_pred$frequency, 
                  df_4_pred$recent_frequency,
                  df_4_pred$avgamount,
                  df_4_pred$maxamount,
                  log(df_4_pred$recency), 
                  log(df_4_pred$frequency),
                  log(df_4_pred$recent_frequency),
                  log(df_4_pred$avgamount),
                  log(df_4_pred$maxamount),
                  df_4_pred$recency * df_4_pred$frequency)

amount = predict(lasso_4_amount, newdata2, s = best.lambda, type = "class")
df_4_pred$amount = amount[,1]
# summary(df_4_pred)
```




# calculate E and solicit
```{r }
df_4_pred$E_money = df_4_pred$donation * df_4_pred$amount
df_4_pred$solicit <-  ifelse (df_4_pred$E_money >= 2, 1, 0)
output <-  df_4_pred %>% select(contact_id, solicit)
output[is.na(output)] <- 0
output = output[order(output$contact_id), ]
write.table(output, "test", sep = '\t', row.name = F, col.name = F)
```














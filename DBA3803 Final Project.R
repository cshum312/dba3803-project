#open libraries
library(dplyr)
library(tidyr)
library(glmnet)
library(rpart)
library(rattle)
library(ggplot2)
library(forcats)
library(randomForest)

#evaluation function
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

#===========================================
#evaluating without G1 and G2

#set seed
set.seed(123)

#import data, remove G1 G2
data <- read.csv2("student-por.csv", stringsAsFactors = T) %>%
  subset(select = -c(G1, G2))

#split test train
set.seed(123)
split <- sort(sample(nrow(data), nrow(data) * 0.8))
data_train <- data[split, ]
data_test <- data[-split, ]

#demean
data_train[,sapply(data_train, is.numeric)] <- apply(data_train[sapply(data_train, is.numeric)], 2, function(x) scale(x, center = T, scale = F))
data_test[,sapply(data_test, is.numeric)] <- apply(data_test[sapply(data_test, is.numeric)], 2, function(x) scale(x, center = T, scale = F))

#perform 10-fold cv to choose lambda
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)

#convert factors to dummy
x_train <- model.matrix(G3 ~ ., data_train)[, -1]
x_list <- colnames(x_train)

x_test <- model.matrix(G3 ~ ., data_test)[, -1]

#plot ridge
ridge_cv <- cv.glmnet(x_train, data_train$G3, alpha = 0, lambda = lambdas_to_try, standardize = F, nfolds = 10)
plot(ridge_cv)
print(ridge_cv)

#plot lasso
lasso_cv <- cv.glmnet(x_train, data_train$G3, alpha = 1, lambda = lambdas_to_try, standardize = F, nfolds = 10)
plot(lasso_cv)
print(lasso_cv)

#find min of ridge and lasso
min_ridge <- ridge_cv$lambda.min
min_lasso <- lasso_cv$lambda.min
paste('Lambda Min (Ridge): ', min_ridge)
paste('Lambda Min (Lasso): ', min_lasso)

#plot ridge coefficients
ridge_res <- glmnet(x_train, data_train$G3, alpha=0, lambda=lambdas_to_try, standardize=F)
plot(ridge_res, xvar="lambda")

#plot lasso coefficients
lasso_res <- glmnet(x_train, data_train$G3, alpha=1, lambda=lambdas_to_try, standardize=F)
plot(lasso_res, xvar="lambda")

#using min lambda to find coefficients (ridge)
ridge_coef <- glmnet(x_train, data_train$G3, alpha=0, lambda=min_ridge, standardize=F)
ridge_list <- ridge_coef[["beta"]]@x
ridge_index <- ridge_coef[["beta"]]@i + 1

ridge_var <- as.data.frame(x_list[ridge_index]) %>%
  mutate(value = ridge_list) %>%
  rename(coef = 1)
print(ridge_var)

#using min lambda to find coefficients (lasso)
lasso_coef <- glmnet(x_train, data_train$G3, alpha=1, lambda=min_lasso, standardize=F)
lasso_list <- lasso_coef[["beta"]]@x
lasso_index <- lasso_coef[["beta"]]@i + 1

lasso_var <- as.data.frame(x_list[lasso_index]) %>%
  mutate(value = lasso_list) %>%
  rename(coef = 1)
print(lasso_var)

#convert dummy matrix to df
y_train <- as.data.frame(model.matrix( ~ ., data_train)) %>%
  subset(select = -c(1))
y_test <- as.data.frame(model.matrix( ~ ., data_test)) %>%
  subset(select = -c(1))

#regression tree
set.seed(123)
tree <- rpart(G3 ~ ., method="anova", data=y_train, minsplit = 50)
fancyRpartPlot(tree)

printcp(tree)

#plot variable importance (regression tree)
tree_vi <- data.frame(imp = tree$variable.importance) %>% 
  rownames_to_column() %>% 
  rename("variable" = rowname) %>% 
  arrange(imp) %>%
  mutate(variable = fct_inorder(variable))

ggplot(tree_vi) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#plot variable importance (splitters only)
splitters <- tree$frame$var[tree$frame$var != "<leaf>"]
split_vi <- tree_vi[tree_vi$variable %in% splitters, ]

ggplot(split_vi) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#random forest regression
set.seed(123)
rf <- randomForest(G3 ~ . , data = y_train, ntree = 500, mtry=5,
                   keep.forest=T, importance=T)

#plot variable importance (random forest regression)
rf_vi <- data.frame(imp = rf$importance) %>%
  subset(select = -c(2)) %>%
  rownames_to_column() %>% 
  rename("variable" = rowname, "imp" = imp..IncMSE) %>% 
  arrange(imp) %>%
  mutate(variable = fct_inorder(variable))

ggplot(rf_vi) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#evaluation table
ridge_trainpred <- predict(ridge_cv, x_train, s = min_ridge, type = "response")
lasso_trainpred <- predict(lasso_cv, x_train, s = min_lasso, type = "response")
tree_trainpred <- predict(tree, y_train, type = "vector")
rf_trainpred <- predict(rf, y_train, type = "response")

dfeval <- data.frame(matrix(ncol = 2, nrow = 4))
colnames(dfeval) <- c("RMSE", "Rsquare")
rownames(dfeval) <- c("Ridge", "Lasso", "Tree", "RF")
dfeval[1, ] <- eval_results(data_train$G3, ridge_trainpred, data_train)
dfeval[2, ] <- eval_results(data_train$G3, lasso_trainpred, data_train)
dfeval[3, ] <- eval_results(data_train$G3, tree_trainpred, data_train)
dfeval[4, ] <- eval_results(data_train$G3, rf_trainpred, data_train)

print(dfeval)

#chosen model performance against test set
ridge_testpred <- predict(ridge_cv, x_test, s = min_ridge, type = "response")

dfchosen <- data.frame(matrix(ncol = 2, nrow = 2))
colnames(dfchosen) <- c("RMSE", "Rsquare")
rownames(dfchosen) <- c("Train", "Test")

dfchosen[1, ] <- eval_results(data_train$G3, ridge_trainpred, data_train)
dfchosen[2, ] <- eval_results(data_test$G3, ridge_testpred, data_test)

print(dfchosen)

#===========================================
#evaluating with G1 and G2

#set seed
set.seed(123)

#import data
data <- read.csv2("student-por.csv", stringsAsFactors = T)

#split test train
set.seed(123)
split <- sort(sample(nrow(data), nrow(data) * 0.8))
data_train <- data[split, ]
data_test <- data[-split, ]

#demean
data_train[,sapply(data_train, is.numeric)] <- apply(data_train[sapply(data_train, is.numeric)], 2, function(x) scale(x, center = T, scale = F))
data_test[,sapply(data_test, is.numeric)] <- apply(data_test[sapply(data_test, is.numeric)], 2, function(x) scale(x, center = T, scale = F))

#perform 10-fold cv to choose lambda
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)

#convert factors to dummy
x_trainall <- model.matrix(G3 ~ ., data_train)[, -1]
x_listall <- colnames(x_trainall)

x_testall <- model.matrix(G3 ~ ., data_test)[, -1]

#plot ridge
ridge_cvall <- cv.glmnet(x_trainall, data_train$G3, alpha = 0, lambda = lambdas_to_try, standardize = F, nfolds = 10)
plot(ridge_cvall)
print(ridge_cvall)

#plot lasso
lasso_cvall <- cv.glmnet(x_trainall, data_train$G3, alpha = 1, lambda = lambdas_to_try, standardize = F, nfolds = 10)
plot(lasso_cvall)
print(lasso_cvall)

#find min of ridge and lasso
min_ridgeall <- ridge_cvall$lambda.min
min_lassoall <- lasso_cvall$lambda.min
paste('Lambda Min (Ridge): ', min_ridgeall)
paste('Lambda Min (Lasso): ', min_lassoall)

#plot ridge coefficients
ridge_resall <- glmnet(x_trainall, data_train$G3, alpha=0, lambda=lambdas_to_try, standardize=F)
plot(ridge_resall, xvar="lambda")

#plot lasso coefficients
lasso_resall <- glmnet(x_trainall, data_train$G3, alpha=1, lambda=lambdas_to_try, standardize=F)
plot(lasso_resall, xvar="lambda")

#using min lambda to find coefficients (ridge)
ridge_coefall <- glmnet(x_trainall, data_train$G3, alpha=0, lambda=min_ridgeall, standardize=F)
ridge_listall <- ridge_coefall[["beta"]]@x
ridge_indexall <- ridge_coefall[["beta"]]@i + 1

ridge_varall <- as.data.frame(x_listall[ridge_indexall]) %>%
  mutate(value = ridge_listall) %>%
  rename(coef = 1)
print(ridge_varall)

#using min lambda to find coefficients (lasso)
lasso_coefall <- glmnet(x_trainall, data_train$G3, alpha=1, lambda=min_lassoall, standardize=F)
lasso_listall <- lasso_coefall[["beta"]]@x
lasso_indexall <- lasso_coefall[["beta"]]@i + 1

lasso_varall <- as.data.frame(x_listall[lasso_indexall]) %>%
  mutate(value = lasso_listall) %>%
  rename(coef = 1)
print(lasso_varall)

#convert dummy matrix to df
y_trainall <- as.data.frame(model.matrix( ~ ., data_train)) %>%
  subset(select = -c(1))
y_testall <- as.data.frame(model.matrix( ~ ., data_test)) %>%
  subset(select = -c(1))

#regression tree
set.seed(123)
treeall <- rpart(G3 ~ ., method="anova", data=y_trainall, minsplit = 50)
fancyRpartPlot(treeall)

printcp(treeall)

#plot variable importance (regression tree)
tree_viall <- data.frame(imp = treeall$variable.importance) %>% 
  rownames_to_column() %>% 
  rename("variable" = rowname) %>% 
  arrange(imp) %>%
  mutate(variable = fct_inorder(variable))

ggplot(tree_viall) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                                  size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#plot variable importance (splitters only)
splittersall <- treeall$frame$var[treeall$frame$var != "<leaf>"]
split_viall <- tree_viall[tree_viall$variable %in% splittersall, ]

ggplot(split_viall) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                                   size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#random forest regression
set.seed(123)
rfall <- randomForest(G3 ~ . , data = y_trainall, ntree = 500, mtry=5,
                      keep.forest=T, importance=T)

#plot variable importance (random forest regression)
rf_viall <- data.frame(imp = rfall$importance) %>%
  subset(select = -c(2)) %>%
  rownames_to_column() %>% 
  rename("variable" = rowname, "imp" = imp..IncMSE) %>% 
  arrange(imp) %>%
  mutate(variable = fct_inorder(variable))

ggplot(rf_viall) + geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                                size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() + theme_bw()

#evaluation table
ridge_trainpredall <- predict(ridge_cvall, x_trainall, s = min_ridgeall, type = "response")
lasso_trainpredall <- predict(lasso_cvall, x_trainall, s = min_lassoall, type = "response")
tree_trainpredall <- predict(treeall, y_trainall, type = "vector")
rf_trainpredall <- predict(rfall, y_trainall, type = "response")

dfevalall <- data.frame(matrix(ncol = 2, nrow = 4))
colnames(dfevalall) <- c("RMSE", "Rsquare")
rownames(dfevalall) <- c("Ridge", "Lasso", "Tree", "RF")
dfevalall[1, ] <- eval_results(data_train$G3, ridge_trainpredall, data_train)
dfevalall[2, ] <- eval_results(data_train$G3, lasso_trainpredall, data_train)
dfevalall[3, ] <- eval_results(data_train$G3, tree_trainpredall, data_train)
dfevalall[4, ] <- eval_results(data_train$G3, rf_trainpredall, data_train)

print(dfevalall)

#chosen model performance against test set
ridge_testpredall <- predict(ridge_cvall, x_testall, s = min_ridgeall, type = "response")

dfchosenall <- data.frame(matrix(ncol = 2, nrow = 2))
colnames(dfchosenall) <- c("RMSE", "Rsquare")
rownames(dfchosenall) <- c("Train", "Test")

dfchosenall[1, ] <- eval_results(data_train$G3, ridge_trainpredall, data_train)
dfchosenall[2, ] <- eval_results(data_test$G3, ridge_testpredall, data_test)

print(dfchosenall)

#===============================

#print evaluation lasso G1 G2
print(lasso_cvall)
print(lasso_cv)


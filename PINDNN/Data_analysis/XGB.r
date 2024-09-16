rm(list=ls());
#install.packages("devtools")
#devtools::install_github("SkadiEye/deepTL")
# args1 = commandArgs(TRUE)
# Folder_args = as.numeric(args1[1])
# start.seed = as.numeric(args1[1]) *2399

args <- commandArgs(trailingOnly = TRUE)
# The first argument is the fold number
fold <- as.integer(args[1])

print('XGB')

library(deepTL);
library(randomForest);
library(xgboost);
library(magrittr);
library(glmnet);
library(MASS);
library(e1071);
library(ROCR);
#library(caret);
#library(GGally);


impfile = paste0("/work/users/x/i/xiaoyw/XGB_imp_",fold,".csv");
aucfile = paste0("/work/users/x/i/xiaoyw/XGB_auc_",fold,".csv");
prefile = paste0("/work/users/x/i/xiaoyw/XGB_pre_",fold,".csv");

pvacut = 0.05;

dat = read.csv("/work/users/x/i/xiaoyw/final_R_0423.csv",head=T); #17018
#dat = na.omit(dat)


#dat$PainScore_std[is.na(dat$PainScore_std)] = 0
# dat <- dat[, -c(21, 22, 23, 50, 54:ncol(dat))]
# dat$ETHNICITY_OTHER.SPECIES <- as.integer(rowSums(dat[, c("ETHNICITY_ASIAN...THAI", "ETHNICITY_OTHER.SPECIES")]) > 0)
# dat$ETHNICITY_ASIAN...THAI <- NULL
# set.seed(20220301);
# dat = dat[sample(nrow(dat), 5000),]
#colnames(dat)


#-----------------permfit----------------------
dimsize = dim(dat)[2];
y = factor(dat[,3], labels=c("0", "1"));
x = as.matrix(dat[,4:dimsize]);
dimsizx = dim(x)[2];

#seedused = 5;
#if (seedused == 1)  {
newx = x;
newy = y;
posct = dat[,3];
#}
# if (seedused > 1)  {
#   set.seed(20220301*seedused);
#   index = sample(1:length(y),length(y),replace=T);
#   newx = x[index,];
#   newy = y[index];
#   posct = dat[index,3];
# } 

#### Functions for Evaluation
acc = function(y, x, cut=0.5) mean((y==levels(y)[2]) == (x>cut));
auc = function(y, x) performance(prediction(x, (y == levels(y)[2])*1), "auc")@y.values[[1]];
prauc = function(y, x) performance(prediction(x, (y == levels(y)[2])*1), "aucpr")@y.values[[1]];

#### 0.0 Categorical feature list ####
#colnames(x)
pathwaylist = list(INSURANCE=18:19, ETHNICITY=21:24, MARITAL_STATUS=25:28, ADMISSION_TYPE=29:30, ADMISSION_LOCATION=31:34, Surgery=35:40);
pathwaylist$pre_infection <- 1
pathwaylist$GENDER <- 20
conlist = c(2:17, 41:dimsizx);

#### 0.1 Hyper-parameters ####
n_ensemble = 100; #100
n_perm = 100; #100
new_perm = 200; #100
fold_num = 10; #10
new_fold = 10;
num_sample = 500;
n_epoch = 1000; #1000
n_tree = 1000;
node_size = 3;

esCtrl = list(n.hidden = c(50, 40, 30, 20), activate = "relu", l1.reg = 10**-4, early.stop.det = 1000, n.batch = 30, 
              n.epoch = n_epoch, learning.rate.adaptive = "adam", plot = FALSE);

#### 0.2 Random Shuffle
# seedused == 1
# if (seedused == 1)  
set.seed(20220301);
shuffle = sample(length(y));
oy = rep(0,length(posct));

#### 2. Cross-Validation ####
testnum = round(length(y) / fold_num, 0);

if (fold < fold_num) {
  range = c((1 + (fold - 1) * testnum) : (fold * testnum));
} else {  # fold == fold_num
  range = c((1 + (fold - 1) * testnum) : length(y));
}

# validate1=shuffle[c(1:500)]
# pred1=newx[validate1, ]
# write.csv(pred1, file="pred1data.csv", row.names=FALSE)
#validate = shuffle[1:num_sample];
validate = shuffle[range];
oy[validate] = posct[validate];
trainx = newx[-validate, ];
trainy = newy[-validate];
trainlen = length(trainy);

# Set the size of 'pred' based on the size of the validation set
pred = matrix(NA, length(validate), 1);
nshuffle = sample(trainlen);

parms = list(booster="gbtree", objective="binary:logistic", eta=0.3, gamma=0, max_depth=5, min_child_weight=1, subsample=1, colsample_bytree=1);

#### 2.13 XGBoost ####
xgbtrain = xgb.DMatrix(data=newx[-validate, ],label=posct[-validate]);
xgbtest = xgb.DMatrix(data=newx[validate, ],label=posct[validate]);
xgb_mod = xgb.train(params=parms, data=xgbtrain, nrounds=5, watchlist=list(train=xgbtrain), print_every_n=NULL, maximize=F, eval_metric="error");
#pred[validate, 8] = predict(xgb_mod,xgbtest);
pred[, 1] = predict(xgb_mod,xgbtest);
print('XGBoost model finished')

#### 3. Summary
## 3.3 Observed and predicted

dfpre = data.frame('PosCT'=oy[validate],'XGBoost'=pred[,1]);
write.table(dfpre,file=prefile,append=F,row.names=F,col.names=T,sep=",");

dfauc = data.frame(Method = "XGBoost",
                   Accuracy = round(apply(pred, 2, function(x) acc(y = newy[validate], x)),3),
                   AUC = round(apply(pred, 2, function(x) auc(y = newy[validate], x)),3),
                   PRAUC = round(apply(pred, 2, function(x) prauc(y = newy[validate], x)),3));
write.table(dfauc,file=aucfile,append=F,row.names=F,col.names=T,sep=",")

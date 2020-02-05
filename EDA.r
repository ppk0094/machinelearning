
#install.packages("ROCR",repos = "https://cloud.r-project.org")
library(MASS)
library(caret)
library(glmnet)
library(usdm)
library(leaps)
library(ROCR)
library(e1071)

#Read the data ticdata2000.txt(Training data) into dataframe
train_data <- read.table("ticdata2000.txt",sep = "\t",header = FALSE)
test_data <- read.table("tictgts2000.txt", sep = "\t", header = FALSE)
eval_data <- read.table("ticeval2000.txt",sep = "\t",header = FALSE)

# Displaying first few records of the dataframe
head(train_data)

# The total number of records and the number of columns
dim(train_data)

str(train_data)

# Return the correlations between each of the predictors,
# in descending order of magnitude.
cor_ <- as.data.frame.table(cor(train_data[ , -ncol(train_data)]))
colnames(cor_) <- c("a", "b", "cor")
cor_ <- cor_[cor_$a != cor_$b, ]
cor_ <- cor_[order(abs(cor_$cor), decreasing = TRUE), ]
cor_ <- cor_[seq(1, nrow(cor_), 2), ]
cor_$cor <- round(cor_$cor, 2)
rownames(cor_) <- 1:nrow(cor_)
print(cor_[1:30, ])

counts <- table(train_data$V86)
barplot(counts, main="Caravan policies", 
   xlab="Caravan policies bought",col=c("darkblue","red"),legend = rownames(counts),beside=TRUE)

tic_s <- train_data[train_data$V86==1,]
tic_s$V59 <- as.factor(tic_s$V43)
tic_s$V86 <- as.factor(tic_s$V86)
counts <- table(tic_s$V43,tic_s$V86)
barplot(counts, main="Purchasing Power Class VS Caravan Policy Buyers", 
   xlab="Caravan Policy Buyers",col=c("darkblue","red","green","yellow","blue","orange","black","pink"),legend = rownames(counts),beside=TRUE)

cor(train_data[,c(44:86)])

tic_s <- train_data[train_data$V86==1,]
tic_s$V59 <- as.factor(tic_s$V68)
tic_s$V86 <- as.factor(tic_s$V86)
counts <- table(tic_s$V68,tic_s$V86)
barplot(counts, main="Number of car policies VS Caravan Policy Buyers", 
   xlab="Caravan Policy Buyers",col=c("darkblue","red","green","yellow"),legend = rownames(counts),beside=TRUE)

zerovar <- nearZeroVar(train_data)

train_new <- train_data[,-c(zerovar)]

train_h <- train_new
train_h$V86 <- as.factor(train_h$V86)
x <- c()
j = 0
for (i in (1:43)){
y <- paste(c("V", i), collapse = "")
train_h[,y] <- as.factor(train_new[,y])
tic <- table(train_h[,y],train_h$V86)
chi2 = chisq.test(tic)
    
if( chi2$p.value > 0.05) {
    j <- j + 1
    x[j] <- y
}
    }

train_h <- train_new[,!names(train_new) %in% c(x)]

k <- findCorrelation(cor(train_h[,!names(train_h) %in% c("V86")]), cutoff = 0.7, verbose = FALSE)

train_k <- train_h[,-c(k)]

train_k$V86 <- as.factor(train_k$V86)
train_lasso <- train_k

xmat <- model.matrix(V86 ~ ., data = train_lasso)

xmat <- xmat[,-1]

cv.lasso <- cv.glmnet(xmat, train_lasso$V86 , alpha = 1,family="binomial")

plot(cv.lasso)

bestlam <- cv.lasso$lambda.min
bestlam

fit.lasso <- glmnet(xmat, train_lasso$V86, alpha = 1,family="binomial")
predict(fit.lasso, s = bestlam, type = "coefficients")[1:29, ]

train_o <- train_k[,!names(train_k) %in% c("V5","V19","V13","V23","V24","V25","V26","V35","V39")] #V27,38

vifcor(cor(train_o[,-ncol(train_o)]),th=0.7)

train_o <-train_o[,!names(train_o) %in% c("V42","V30","V37","V28","V43","V59","V10")]

res.glm <- glm(V86~.,data = train_o, family = "binomial")
summary(res.glm)

regfit.full <- regsubsets(V86 ~ ., data = train_o, nvmax = ncol(train_o))
reg.summary<-summary(regfit.full)
reg.summary

names(reg.summary)

plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
mincp = which.min(reg.summary$cp)
points(mincp, reg.summary$cp[mincp], col = "red", cex = 2, pch = 20)
mincp

regfit.bwd <- regsubsets(V86 ~ ., data = train_o, nvmax = 15, method = "backward")
reg.summary.bwd <- summary(regfit.bwd)
reg.summary.bwd

par(mfrow = c(2, 2))
plot(reg.summary.bwd$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary.bwd$cp), reg.summary.bwd$cp[which.min(reg.summary.bwd$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary.bwd$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary.bwd$bic), reg.summary.bwd$bic[which.min(reg.summary.bwd$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary.bwd$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary.bwd$adjr2), reg.summary.bwd$adjr2[which.max(reg.summary.bwd$adjr2)], col = "red", cex = 2, pch = 20)
plot(reg.summary.bwd$rss, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
mtext("Plots of C_p, BIC, adjusted R^2 and RSS for backward stepwise selection", side = 3, line = -2, outer = TRUE)

res.glm <- glm(V86~.,data = train_o, family = "binomial")
summary(res.glm)

res.glm <- glm(V86~V68+V16+V44+V32+V7+V8+V17+V21+V40+V22+V29,data = train_o, family = "binomial")
summary(res.glm)

res.glm <- glm(V86~V68+V16+V44+V32+V7+V8+V17+V21+V40+V22,data = train_o, family = "binomial")
summary(res.glm)

res.glm <- glm(V86~V68+V16+V44+V32+V7+V8+V17+V21+V40,data = train_o, family = "binomial")
summary(res.glm)

res.glm <- glm(V86~V68+V16+V44+V32+V7+V17+V21+V40,data = train_o, family = "binomial")
summary(res.glm)

res.glm <- glm(V86~V68+V16+V44+V32+V7+V21+V40,data = train_o, family = "binomial")
summary(res.glm)

smp_size <- floor(0.75 * nrow(train_o))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train_o)), size = smp_size)

train <- train_o[train_ind, ]
test <- train_o[-train_ind, ]
actual_V86 <- test
test$V86 <- NULL

res.glm <- glm(V86~V68+V16+V44+V32+V7+V17+V21+V40,data = train, family = "binomial")
probs2 <- predict(res.glm, test, type = "response")
pred.glm2 <- rep(0, length(probs2))
pred.glm2[probs2 > 0.5] <- 1
confusionMatrix(table(pred.glm2, actual_V86$V86),positive='1')

# Plot ROC and AUC for LR
probs <- probs2
LRPred <- prediction(probs, actual_V86$V86)
LRPerf <- performance(LRPred, "tpr", "fpr")
plot(LRPerf, colorize=TRUE)
abline(a=0, b=1, lty=2, lwd=3, col="black")

performance(LRPred, "auc")

nb_default <- naiveBayes(V86~V68+V16+V44+V32+V7+V17+V21+V40, data=train)
probs2 <- predict(nb_default,test)
probs_raw <- data.frame(predict(nb_default,test,type="raw"))
tp <- table(probs2, actual_V86$V86)
confusionMatrix(table(probs2, actual_V86$V86),positive='1')

# Plot ROC and AUC for LR
probs <- probs_raw
LRPred <- prediction(probs$X1, actual_V86$V86)
LRPerf <- performance(LRPred, "tpr", "fpr")
plot(LRPerf, colorize=TRUE)
abline(a=0, b=1, lty=2, lwd=3, col="black")

performance(LRPred, "auc")

fit.lda = lda(V86 ~ V68+V16+V44+V32+V7+V17+V21+V40, data = train)
pred.lda = predict(fit.lda, test)
confusionMatrix(pred.lda$class, actual_V86$V86, positive="1")

# Plot ROC and AUC for LR

probs <- pred.lda
LRPred <- prediction(probs$posterior[,2], actual_V86$V86)
LRPerf <- performance(LRPred, "tpr", "fpr")
plot(LRPerf, colorize=TRUE)
abline(a=0, b=1, lty=2, lwd=3, col="black")

performance(LRPred, "auc")

nb_default <- naiveBayes(V86~V68+V16+V44+V32+V7+V17+V21+V40, data=train_o)
default_pred <- predict(nb_default, eval_data, type="raw")
prob1<- data.frame(default_pred)
prob1$index <- c(1:4000)
prob_o<- prob1[order(-prob1$X1),]
for (i in c(1:800)){
    prob_o$caravan[i] <- 1
}
for (i in c(801:4000)){
    prob_o$caravan[i] <- 0
}


prob_new <- prob_o[order(prob_o$index),]
confusionMatrix(table(prob_new$caravan, test_data$V1,dnn=c("Prediction","Actual")),positive='1')

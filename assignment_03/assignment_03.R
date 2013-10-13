# Set working directory
setwd("C:/Users/Iain/my_homework/assignment_03")

# Open required libraries
library(glmnet)
library(DAAG)
library(tm)
library(ggplot2)
#library(qdap)
Sys.setenv(JAVA_HOME="")
library(RWeka)

# define Mean Absolute Error
mae <- function(one, other) {
  return(mean(abs(one - other)))
} # Make a note about R scoping!

# define Error From Fold
error_from_fold <- function(n) {
  last.col <- ncol(trainer)
  trainx <-as.matrix(subset(trainer, n != fold)[,-((last.col - 1):last.col)])
  trainy <-as.matrix(subset(trainer, n != fold)[,(last.col - 1)])
  model <- glmnet(trainx, log(trainy), lambda=best.lambda)
  testx <- as.matrix(subset(trainer, n == fold)[,-((last.col - 1):last.col)])
  testy <-as.matrix(subset(trainer, n == fold)[,(last.col - 1)])
  error <- mae(exp(predict(model, testx)), testy) 
  return(error)
}

####################################################
# STEP 0 - Read in training data and review it
####################################################
alltrain <- read.csv("train.csv")

# check distribution of variable to be predicted - SalaryNormalized
qplot(SalaryNormalized, data=alltrain)

# distribution of SalaryNormalized has noticeable positive skew - check log transform
qplot(log(SalaryNormalized), data=alltrain)
# that looks more like a normal distribution

# check distributions of ContractType and ContractTime
qplot(ContractType, data=alltrain)
qplot(ContractTime, data=alltrain)
# seems that most values are missing for each of these variables
# of those recorded there are far more full_time than part_time and far more permanent than contract

# check how salary distribution relates to ContractType and ContractTime
qplot(log(SalaryNormalized), data=alltrain, fill = ContractType)
qplot(log(SalaryNormalized), data=alltrain, fill = ContractTime)
qplot(log(SalaryNormalized), ContractType, data=alltrain, geom = "jitter", colour = ContractTime)
qplot(log(SalaryNormalized), ContractTime, data=alltrain, geom = "jitter", colour = ContractType)

# some other plots
qplot(Category, data=alltrain)
qplot(LocationNormalized, data=alltrain)
qplot(Company, data=alltrain)


####################################################
# STEP 1 - Split data into training and test sets
####################################################
# Randomly select fold assignments for n-fold cross-validation
set.seed(42)
alltrain$fold <- sample(1:10, nrow(alltrain), replace=TRUE)

####################################################
# STEP 2 -  Build a simple linear regression using the available categorical variables
####################################################

# Predict using all training set for log of Salary Normalized using all cat variables and convert back from log for MAE
fit1 <- lm(log(SalaryNormalized) ~ Category + LocationNormalized + Company + ContractType + ContractTime + SourceName, data = alltrain)
summary(fit1)
mae(exp(fitted(fit1)), alltrain$SalaryNormalized)
# R2 = 0.5554 MAE = 6998.585

# try removing SourceName
fit2 <- lm(log(SalaryNormalized) ~ Category + LocationNormalized + Company + ContractType + ContractTime, data = alltrain)
summary(fit2)
mae(exp(fitted(fit2)), alltrain$SalaryNormalized)
# R2 = 0.5496 MAE = 7069.786 - worse than fit1

# put SourceName back in and remove ContractTime
fit3 <- lm(log(SalaryNormalized) ~ Category + LocationNormalized + Company + ContractType + SourceName, data = alltrain)
summary(fit3)
mae(exp(fitted(fit3)), alltrain$SalaryNormalized)
# R2 = 0.5529 MAE = 7031.836 - not as good as fit1

# put ContractTime back in and remove ContractType
fit4 <- lm(log(SalaryNormalized) ~ Category + LocationNormalized + Company + ContractTime + SourceName, data = alltrain)
summary(fit4)
mae(exp(fitted(fit4)), alltrain$SalaryNormalized)
# R2 = 0.541 MAE = 7114.696 - not as good as fit1

# put ContractType back in and remove Company
fit5 <- lm(log(SalaryNormalized) ~ Category + LocationNormalized + ContractType + ContractTime + SourceName, data = alltrain)
summary(fit5)
mae(exp(fitted(fit5)), alltrain$SalaryNormalized)
# R2 = 0.4123 MAE = 8462 - much worse than fit1

# put Company back in and remove Location
fit6 <- lm(log(SalaryNormalized) ~ Category + Company + ContractType + ContractTime + SourceName, data = alltrain)
summary(fit6)
mae(exp(fitted(fit6)), alltrain$SalaryNormalized)
# R2 =  0.4633 MAE =  7716.953 - worse than fit1

# put Location back in and remove Category
fit7 <- lm(log(SalaryNormalized) ~ LocationNormalized + Company + ContractType + ContractTime + SourceName, data = alltrain)
summary(fit7)
mae(exp(fitted(fit7)), alltrain$SalaryNormalized)
# R2 = 0.544 MAE = 7062 - not as good as fit1

## Conclusion - removing any of variables increases error of predictions on whole training data set
## Removing Company and Location had the most (negative) effect

fit8 <- lm(log(SalaryNormalized) ~ LocationNormalized + Company + Category + ContractType:ContractTime + SourceName, data = alltrain)
summary(fit8)
mae(exp(fitted(fit8)), alltrain$SalaryNormalized)
# R2 = 0.556 MAE = 6993.274 slightly better than fit 1

fit9 <- lm(log(SalaryNormalized) ~ LocationNormalized + Company + Category + ContractType * ContractTime + SourceName, data = alltrain)
summary(fit9)
mae(exp(fitted(fit9)), alltrain$SalaryNormalized)
# R2 = 0.556 MAE = 6993.274 , same as fit8

# Tried other interactions of variables with more levels but could not run them with current computing resources
# Decide to use all variables as they all seem to have some predictive power

# Now format training and tests set consistently for these dummy categorical variables:

# Read in separate train and test files
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Combine them for the column(s) we want to use as predictors in our model
all <- rbind(train[, c("Category","Company", "LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F],
             test[, c("Category", "Company","LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F])

# Explicitly construct all the dummy columns for the Category variable
allx <- model.matrix(~LocationNormalized + Company + Category + ContractType:ContractTime + SourceName, data=all)

# Split out the training and test data, adding in the response variable as well
trainer <- cbind(as.data.frame(allx[1:10000,]), train[,"SalaryNormalized", drop=F])
tester <- cbind(as.data.frame(allx[10001:15000,]), data.frame(SalaryNormalized=NA))

############################################################
# STEP 3/5 -  Use cross validation, DAAG and glmnet
############################################################

#modelDAAG <- CVlm(df = trainer, form.lm = formula(log(SalaryNormalized) ~ .), m=10)
#pred <- predict(modelDAAG, trainer)

# DAAG was very slow and ouput was a bit confusing

# prepare data for glmnet
x <- as.matrix(trainer[-3928])
y <- as.matrix(trainer[3928])

# run glmnet
glmcv.model1 <- cv.glmnet(x, log(y), type.measure="mae")
print(glmcv.model1) # MAE with best lambda value is 0.2706664 (log of SalaryNormalized)

# MAE in the cv.glmnet model is the log value, unsure how to back transform the errors
# so check predictions against training set
test.predict.reg <- predict(glmcv.model, x)
mae(exp(test.predict.reg), y) # MAE for training set is 8095 (SalaryNormalized)

# Try own cv using bestlambda value from glmnet
best.lambda <- glmcv.model1$lambda.min
set.seed(42)
trainer$fold <- sample(1:10, nrow(trainer), replace=TRUE)
fold.errors <- sapply(1:10, error_from_fold)
print(fold.errors)
print(mean(fold.errors)) # MAE 8748 (SalaryNormalized)

############################################################
# STEP 4 -  Location tree
############################################################

# Intended to use the broader regions in the location tree instead of LocationNormalized
# but did not manage to find a good way to lookup this value from LocationRaw or
# LocationNormalized

#loc.tree <- read.csv("Location_Tree2.csv", header = FALSE)
#names(loc.tree) <- c("Country", "Region", "County", "Town")
#all.loc <- rbind(train[, "LocationRaw", drop=F],
#                 test[, "LocationRaw", drop=F])
#write.csv(all.loc, "all_loc.csv")
#all.loc.txt <- read.delim("all_loc.txt", header = FALSE)
#loc1 <- all.loc.txt[2]
#names(loc1) <- "County"



####################################################
# STEP 6 -  Text Features
####################################################

# bring in Title field and once again process training and test set together
all <- rbind(train[, c("Title", "Category","Company", "LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F],
             test[, c("Title", "Category", "Company","LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F])

# create corpus from $Title
src <- DataframeSource(data.frame(all$Title)) # You can use any of the text columns, not just Title.
c <- Corpus(src)

# convert corpus to lower case
c <- tm_map(c, tolower)

# create control list for dtm
control <- list(stopwords = TRUE,
                removePunctuation = TRUE,
                removeNumbers = TRUE,
                minDocFreq = 2)
# Create DocumentTermMatrix 
dtm <- DocumentTermMatrix(c, control)

# Create columns for some frequent terms used in the Title field
text_data <- cbind(allx, as.matrix(dtm[,'senior']), as.matrix(dtm[,'junior']), as.matrix(dtm[,'principal']),
                   as.matrix(dtm[,'manager']), as.matrix(dtm[,'nurse']), as.matrix(dtm[,'worker']),
                   as.matrix(dtm[,'assistant']), as.matrix(dtm[,'chef']), as.matrix(dtm[,'software']),
                   as.matrix(dtm[,'business']), as.matrix(dtm[,'marketing']), as.matrix(dtm[,'executive']),
                   as.matrix(dtm[,'administrator']), as.matrix(dtm[,'teacher']), as.matrix(dtm[,'account']),
                   as.matrix(dtm[,'engineer']), as.matrix(dtm[,'analyst']), as.matrix(dtm[,'sales']))

# Split out the training and test data, adding in the response variable as well
trainer <- cbind(as.data.frame(text_data[1:10000,]), train[,"SalaryNormalized", drop=F])
tester <- cbind(as.data.frame(text_data[10001:15000,]), data.frame(SalaryNormalized=NA))
sal.col <- ncol(trainer)

# Prepare data for cv.glmnet
x <- as.matrix(trainer[,-sal.col])
y <- as.matrix(trainer[,sal.col])

# run cv.glmnet
glmcv.model2 <- cv.glmnet(x, log(y), type.measure="mae")
print(glmcv.model2) # MAE 0.2469484 (log of SalaryNormalized)

# check how model does on training set
test.predict.reg <- predict(glmcv.model2, x)
mae(exp(test.predict.reg),y) # MAE 7514 (SalaryNormalized)

# obtain best lambda value from cv.glmnet model
best.lambda <- glmcv.model2$lambda.min

# do some cv using glmnet and best lambda
set.seed(42)
trainer$fold <- sample(1:10, nrow(trainer), replace=TRUE)
fold.errors <- sapply(1:10, error_from_fold)
print(fold.errors)
print(mean(fold.errors)) # MAE 8079 (SalaryNormalized)

#####################################################################
# Try n-grams from $Title
######################################################################
# clean up corpus
c1 <- tm_map(c, removeWords, stopwords('english'))
c2 <- tm_map(c1, removePunctuation)
c3 <- tm_map(c2, removeNumbers)

# use NGramTokenizer function from RWeka package
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Create dtm using tokenizer for 2-grams
dtm <- DocumentTermMatrix(c3, control = list(tokenize = BigramTokenizer, minDocFreq = 50))

# Get list of 2-grams which appear in at least 15 documents
most.freq <- findFreqTerms(dtm, 15)

# Convert tomatrix to use with glmnet
ngram_data <- as.matrix(dtm[,most.freq])

# Combine with categorical variables
text_data <- cbind(allx, ngram_data)

# Split out the training and test data, adding in the response variable as well
trainer <- cbind(as.data.frame(text_data[1:10000,]), train[,"SalaryNormalized", drop=F])
tester <- cbind(as.data.frame(text_data[10001:15000,]), data.frame(SalaryNormalized=NA))
sal.col <- ncol(trainer)

# Prepare data for cv.glmnet
x <- as.matrix(trainer[,-sal.col])
y <- as.matrix(trainer[,sal.col])

# run cv.glmnet
glmcv.model3 <- cv.glmnet(x, log(y), type.measure="mae")
print(glmcv.model3) # MAE 0.2257973 (log SalaryNormalized)

# check how model does on training set
test.predict.reg <- predict(glmcv.model3, x)
mae(exp(test.predict.reg),y) # MAE 6786 (SalaryNormalized)

# obtain best lambda value from cv.glmnet model
best.lambda <- glmcv.model3$lambda.min

# do some cv using glmnet and best lambda
set.seed(42)
trainer$fold <- sample(1:10, nrow(trainer), replace=TRUE)
ngram.errors <- sapply(1:10, error_from_fold)
print(ngram.errors)
print(mean(ngram.errors)) # MAE 7472


################################################################
# Make Prediction file
################################################################
# train model using entire training set
best.lambda <- 0.001914475  # as determined using cv.glmnet for this model
x <- as.matrix(trainer[,-sal.col])
y <- as.matrix(trainer[,sal.col])
finalmodel <- glmnet(x, log(y), lambda=best.lambda)

tester.x <- as.matrix(tester)

predictions <- exp(predict(finalmodel, tester.x))
# What are these predictions going to be?

# Put the submission together and write it to a file
submission <- data.frame(Id=test$Id, Salary=predictions)
names(submission) <- c("Id","Salary")
write.csv(submission, "my_submission.csv", row.names=FALSE)


########################################################################
# Step 7 - use 50k training set
########################################################################
train <- read.csv("train_50k.csv")
test <- read.csv("test.csv")

# Combine them for the column(s) we want to use as predictors in our model
all <- rbind(train[, c("Title", "Category","Company", "LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F],
             test[, c("Title", "Category", "Company","LocationNormalized", "ContractType","ContractTime","SourceName"), drop=F])

# Explicitly construct all the dummy columns for the Category variable
allx <- model.matrix(~LocationNormalized + Company + Category + ContractType:ContractTime + SourceName, data=all)

# create corpus from $Title
src <- DataframeSource(data.frame(all$Title)) # You can use any of the text columns, not just Title.
corp <- Corpus(src)

# clean up corpus
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords('english'))
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeNumbers)

# use NGramTokenizer function from RWeka package
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Create dtm using tokenizer for 2-grams
dtm <- DocumentTermMatrix(corp, control = list(tokenize = BigramTokenizer, minDocFreq = 50))

# Get list of 2-grams which appear in at least 15 documents
most.freq <- findFreqTerms(dtm, 15)

# Convert tomatrix to use with glmnet
ngram_data <- as.matrix(dtm[,most.freq])

# Combine with categorical variables
text_data <- cbind(allx, ngram_data)

# Split out the training and test data, adding in the response variable as well
trainer <- cbind(as.data.frame(text_data[1:50000,]), train[,"SalaryNormalized", drop=F])
tester <- cbind(as.data.frame(text_data[50001:55000,]), data.frame(SalaryNormalized=NA))
sal.col <- ncol(trainer)

# Prepare data for cv.glmnet
x <- as.matrix(trainer[,-sal.col])
y <- as.matrix(trainer[,sal.col])

# run cv.glmnet4
glmcv.model4 <- cv.glmnet(x, log(y), type.measure="mae")
print(glmcv.model4) # MAE (log SalaryNormalized)

# check how model does on training set
test.predict.reg <- predict(glmcv.model4, x)
mae(exp(test.predict.reg),y) # MAE (SalaryNormalized)

# obtain best lambda value from cv.glmnet model
best.lambda <- glmcv.model4$lambda.min

# do some cv using glmnet and best lambda
set.seed(42)
trainer$fold <- sample(1:10, nrow(trainer), replace=TRUE)
50k.errors <- sapply(1:10, error_from_fold)
print(50k.errors)
print(mean(50.errors)) # MAE 
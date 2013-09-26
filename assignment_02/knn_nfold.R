# Load required library
library(class)
library(ggplot2)

#################################################
# PREPROCESSING
#################################################

data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set (note: could
# alternatively use neg indices on column index in knn call)

#################################################
# TRAIN/TEST SPLIT
#################################################

set.seed(1)         # initialize random seed for consistency
# NOTE -- run for various seeds --> need for CV!

train.pct <- 0.7    # pct of data to use for training set
N <- nrow(data)     # total number of records (150)

train.index <- sample(1:N, train.pct * N)       # random sample of records (training set)

train.data <- data[train.index, ]       # perform train/test split
test.data <- data[-train.index, ]       # note use of neg index...different than Python!

train.labels <- as.factor(as.matrix(labels)[train.index, ])     # extract training set labels
test.labels <- as.factor(as.matrix(labels)[-train.index, ])     # extract test set labels

#################################################
# APPLY MODEL
#################################################

err.rates <- data.frame()       # initialize results object

max.k <- 100
for (k in 1:max.k)              # perform fit for various values of k
{
  knn.fit <- knn(train = train.data,          # training set
                 test = test.data,           # test set
                 cl = train.labels,          # true labels
                 k = k                       # number of NN to poll
  )
  
  cat('\n', 'k = ', k, ', train.pct = ', train.pct, '\n', sep='')     # print params
  print(table(test.labels, knn.fit))          # print confusion matrix
  
  this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err
  err.rates <- rbind(err.rates, this.err)     # append err to total results
}

#################################################
# OUTPUT RESULTS
#################################################

results <- data.frame(1:max.k, err.rates)   # create results summary data frame
names(results) <- c('k', 'err.rate')        # label columns of results df

# create title for results plot
title <- paste('knn results (train.pct = ', train.pct, ')', sep='')

# create results plot
results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_point() + geom_line()
results.plot <- results.plot + ggtitle(title)

# draw results plot (note need for print stmt inside script to draw ggplot)
print(results.plot)

#################################################
# HOMEWORK
#################################################

knn.nfold <- function(n, k, data.set=data, label.set=labels) 
{ 
  # create n-fold partition of dataset 
  data.pool <- data.set                            # initialize data pool to be partitioned
  fold.list <- list()                              # initialize object to store each fold
  N <- nrow(data.set)                              # obtain total no. of rows in dataset
  r = N%%n                                         # calculate remainder of total rows/no. of folds,these r remainder rows will be distributed between folds
  for (x in 1:n)                                   # create partition for n folds
  {
      ifelse(x<=r,req.rows <-(floor(N/n)+1), req.rows <- floor(N/n))  # condition to distribute the remainder rows to the first r folds    
      set.seed(x)                                        # initialize random seed for repeatability, changes for each fold
      fold.index <- sample(1:nrow(data.pool), req.rows)  # obtain random sample of indices from data pool
      new.fold <- data.pool[fold.index, ]                # initialize fold from random indices
      fold.list[[x]] <- new.fold                         # add new fold to list of folds
      data.pool <- data.pool[-fold.index,]               # remove new fold data from data pool
    
  }
  # perform knn classification n times 
  fold.errs <- data.frame()       # initialize results object
  for (x in 1:n)                  # perform fit for n-folds
  {
    test.index <- as.integer(rownames(fold.list[[x]]))           # obtain indices of the test set (current fold)
    test.labels <- as.factor(as.matrix(label.set)[test.index, ]) # extract test labels from list of all labels
    knn.fit <- knn(train = data.set[-test.index,],               # training set (excludes current test fold)
                   test = fold.list[[x]],                        # test set (current fold)
                   cl = label.set[-test.index],                  # true labels, excluding test fold labels
                   k = k                                         # number of NN to poll
    )
    levels(test.labels) <- levels(label.set)                     # ensure all levels are present in test labels to compare with knn.fit
    this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err
    cat('\n', '    k = ', k, ', fold ', x, ' of ', n,': error = ', this.err, sep='')     # print params and fold error
        
    fold.errs <- rbind(fold.errs, this.err)     # append err to total results
  }
  # n-fold generalization error = average over all iterations
  names(fold.errs)  <- "error"                                # name column in results object
  mean.error = mean(fold.errs$error)                          # calculate mean error over all results
  cat('\n\n', 'KNN Classification with k = ', k, '\n')        # print params
  cat('\n', 'Mean generalization error for', n,'- fold cross validation = ', mean.error, '\n\n')   # print mean error
  return (mean.error)
}

# test the k value range of interest (7-15) from earlier plot with 5, 10 and 20 - fold cross validation
test.fold = data.frame()
for (n in c(5, 10, 20))
{  
  for (k in 7:15)
  {  
    result <- c(k, n, knn.nfold(n, k))
    test.fold <- rbind(test.fold, result)
  }
}
names(test.fold) <- c("k", "n", "error")
test.fold

# create test results plot
test.fold$n <- as.character(test.fold$n)    
test.plot <- ggplot(test.fold, aes(x=k, y=error, group = n, colour = n)) + geom_point() + geom_line()
test.plot <- test.plot + ggtitle("Generalization Error v K for 10-, 15- and 20-Fold Cross Validation")

# draw results plot
print(test.plot)
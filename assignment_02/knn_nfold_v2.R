# Load required library
library(class)
library(ggplot2)

#################################################
# FUNCTIONS
################################################

# Function to return random indices to use for training set

cv.indices <- function (train.pct, seed.val, data.set=data)
{
  set.seed(seed.val)         # initialize random seed for consistency
  # NOTE -- run for various seeds --> need for CV!
  
  N <- nrow(data.set)     # total number of records (150)
  
  train.index <- sample(1:N, train.pct * N)       # random sample of records (training set)
  return(train.index)
  
}

# function to carry out knn n-times using specified k and % of data used for training
# returns mean generalization error

knn.nfold <- function (k, n, train.pct, data.set=data, label.set=labels) 
{
  err.rates <- data.frame()
  for (x in 1:n)
  {
    train.index <- cv.indices(train.pct, x, data.set)            # function call to generate random training set indices
    test.labels <- label.set[-train.index]                       # labels for test set
    knn.fit <- knn(train = data.set[train.index,],               # training set (excludes current test fold)
                   test = data.set[-train.index,],               # test set (current fold)
                   cl = label.set[train.index],                  # true labels, excluding test fold labels
                   k = k                                         # number of NN to poll
    )
    this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err
    print(table(test.labels, knn.fit))                               # print confusion matrix
    cat('\n', '    k = ', k, ', fold ', x, ' of ', n,': error rate = ', this.err,'\n\n', sep='')     # print params and fold error
    err.rates <- rbind(err.rates, this.err)     # append err to total results
  }
  names(err.rates)  <- "error"                                # name column in results object
  mean.error = mean(err.rates$error)                          # calculate mean error over all results
  cat('\n', 'KNN Classification with k = ', k, '\n')          # print params
  cat('\n', 'Mean generalization error rate using', n,'- fold cross validation = ', mean.error, '\n\n')   # print mean error
  return (mean.error)
}

#################################################
# PREPROCESSING #
#################################################

data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set (note: could
# alternatively use neg indices on column index in knn call)

#################################################
# test the k value range of interest (7-15) with 5-, 10- and 20-fold cross validation
##################################################
test.fold = data.frame()
for (n in c(5, 10, 20))
{  
  for (k in 7:15)
  {  
    result <- c(k, n, knn.nfold(k, n, 0.7))
    test.fold <- rbind(test.fold, result)
  }
}
names(test.fold) <- c("k", "n", "error")
test.fold

# create test results plot
test.fold$n <- as.character(test.fold$n)    
test.plot <- ggplot(test.fold, aes(x=k, y=error, group = n, colour = n)) + geom_point() + geom_line()
test.plot <- test.plot + ggtitle("Generalization Error v K for 10-, 15- and 20-Fold Cross Validation")
test.plot <- test.plot + ylab("Mean Generalization Error") + xlab("K-value")
# draw results plot
print(test.plot)


